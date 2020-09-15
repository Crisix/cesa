import gc
import logging
from collections import defaultdict
from copy import copy

import numpy as np
import torch
from torch.nn.functional import mse_loss

from config import model_config, USE_GRADIENTS_FOR_SENTENCE_RELEVANCE
from search_utils.Example import Example
from search_utils.Query import Query
from search_utils.Result import Result
from search_utils.Sentence import Sentence
from search_utils.Statistics import Statistics
from search_utils.expand_sentence import expand_sentence, expand_sentence_neighbour
from search_utils.merge_examples import generate_merged_examples
from search_utils.schedule_words import Edit
from search_utils.split_sentence import get_sentence_word_mapping
from search_utils.sufficient_check import examples_are_sufficient

logger = logging.getLogger(__name__)
debug = logger.debug


def generate_counterfactuals(sentence: str, query: Query) -> Result:
    if query.allow_splitting:
        return _gen_cf_ex_long(sentence, query)
    else:
        return _gen_cf_ex(sentence, query)


def _gen_cf_ex(text: str, query) -> Result:
    """

    Parameters
    ----------
    text: str
    query : search_helper.classes.Query.Query

    Returns
    -------

    """
    gc.collect()
    torch.cuda.empty_cache()

    stats = Statistics(original_sentence=text)
    stats.total_duration.resume()
    stats.query = query

    text = model_config.tokenizer.clean_up_tokenization(text)
    text = Sentence(text)
    original_cls = text.calc_sentiment()
    stats.original_classification = original_cls

    examples = []
    schedule = text.calc_edit_schedule(query)
    for schedule_idx, (edit_strategy, word_indices) in enumerate(schedule):
        assert isinstance(schedule_idx, int)
        with stats.find_matching_words_duration:
            if edit_strategy == Edit.NEIGHBOURS:
                batch = expand_sentence_neighbour(text.text, query, word_indices, schedule_idx)
            else:
                word_indices, mask_indices = word_indices
                batch = expand_sentence(text.text, word_indices, query, mask_indices, schedule_idx)
            # filtering only 'relevant' makes the found words more extreme
            # relevant_batch = [b for b in batch if abs(original_cls[cls_idx] - b.cls[cls_idx]) > MIN_SENTIMENT_CHANGE]
            relevant_batch = batch

        debug(f"{len(examples)} examples total | {len(batch)} new  for {len(word_indices)} words with {schedule_idx} highest gradient")

        stats.tried_examples += len(batch)
        examples.extend(relevant_batch)

        with stats.merging_duration:
            num_per_group = max(4 - schedule_idx, 1) if schedule_idx < 10 else -1
            merged_examples = generate_merged_examples(text, examples, query, num_per_group)
            examples.extend(merged_examples)
            stats.tried_examples += len(merged_examples)

        results: Result = examples_are_sufficient(examples, query)
        if results.sufficient():
            stats.total_duration.pause()
            assert stats.all_timers_stopped()
            results.stats = stats
            results.query = query
            return results

    results: Result = examples_are_sufficient(examples, query)
    stats.total_duration.pause()
    assert stats.all_timers_stopped()
    results.stats = stats
    results.query = query
    return results


# SPLIT TEXTS BEFORE SEARCHING
def calc_sentence_edit_schedule(query, sw_map, text):
    if USE_GRADIENTS_FOR_SENTENCE_RELEVANCE:
        gradients = text.calc_gradients(query.wanted_cls)
        sent_grad = defaultdict(list)
        for i in range(len(text.words)):
            idx = [a <= i <= b for (a, b) in sw_map].index(True)
            sent_grad[idx].append(gradients[i])
        gradients_per_sentence = [(si, np.linalg.norm(g)) for (si, g) in sent_grad.items()]
        edit_sentence_order = [y[0] for y in sorted(gradients_per_sentence, key=lambda x: x[1], reverse=True)]
    else:
        # use distance to wanted classification for relevance
        dist_to_wanted_cls = []
        for start, stop in sw_map:
            sub = model_config.tokenizer.clean_up_tokenization(" ".join(text.words[start:stop + 1]))
            cls = Sentence(sub).calc_sentiment()
            dst = mse_loss(torch.tensor(cls), torch.tensor(query.wanted_cls, dtype=torch.float32))
            dist_to_wanted_cls.append(dst)
        edit_sentence_order = np.argsort(-np.array(dist_to_wanted_cls))
    return edit_sentence_order


# SPLIT TEXTS BEFORE SEARCHING
def _gen_cf_ex_long(text: str, query) -> Result:
    """

    Parameters
    ----------
    text : str
    query: search_helper.classes.Query.Query

    Returns
    -------
    Result
    """

    gc.collect()
    torch.cuda.empty_cache()

    stats = Statistics(original_sentence=text)
    stats.total_duration.resume()
    stats.query = query
    stats.original_sentence = text

    text = model_config.tokenizer.clean_up_tokenization(text)
    text = Sentence(text)
    stats.original_classification = text.calc_sentiment()

    sw_map = get_sentence_word_mapping(text.text)

    edit_sentence_order = calc_sentence_edit_schedule(query, sw_map, text)

    examples = []
    for sentence_sched_idx, si in enumerate(edit_sentence_order[:query.consider_max_sentences]):
        debug("> subsentence")
        start, stop = sw_map[si]
        sub = model_config.tokenizer.clean_up_tokenization(" ".join(text.words[start:stop + 1]))

        sub_query = copy(query)
        subresult: Result = _gen_cf_ex(sub, sub_query)

        if stats.tried_sentences is None:
            stats.tried_sentences = 0
        stats.tried_sentences += 1
        stats.add(subresult.stats)

        subexample: Example
        best_subexamples = [j[0] for j in subresult.examples]
        debug(f"SUBEXAMPLE SEARCH FOUND {len(best_subexamples)} of {sub_query.num_needed}")
        if len(best_subexamples) < query.num_needed:
            best_subexamples.extend(subresult.rest[:(query.num_needed - len(best_subexamples))])
            debug(f"Added from rest, now {len(best_subexamples)}")

        for subexample in best_subexamples:
            new_sen = list(text.words)
            new_sen[start:stop + 1] = [subexample.sentence]
            new_sen = Sentence(model_config.tokenizer.clean_up_tokenization(" ".join(new_sen)))
            new_cls = new_sen.calc_sentiment()
            # print(np.round(new_cls, 3), new_sen.text)

            new_changes = [(pos + start, dist) for (pos, dist) in subexample.changes]
            e = Example(new_sen.text, new_cls, new_changes,
                        pred_ind=subexample.prediction_indices,
                        sched_ind=subexample.schedule_indices,
                        sent_ind=[sentence_sched_idx])
            examples.append(e)

        with stats.merging_duration:
            debug("> subsentence merge")
            merged_examples = generate_merged_examples(text, examples, query, 1)
            examples.extend(merged_examples)
            stats.tried_examples += len(merged_examples)
            debug("< subsentence merge")

        results: Result = examples_are_sufficient(examples, query)
        if results.sufficient():
            stats.total_duration.pause()
            assert stats.all_timers_stopped()
            results.stats = stats
            results.query = query
            results.sentence_map = sw_map
            return results
        debug("< subsentence")

    results: Result = examples_are_sufficient(examples, query)
    stats.total_duration.pause()
    assert stats.all_timers_stopped()
    results.stats = stats
    results.query = query
    results.sentence_map = sw_map
    return results

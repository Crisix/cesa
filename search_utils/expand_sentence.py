import logging
from typing import Union, List

from config import MBS_DEPTH, MBS_BEAMS
from search_utils.Example import Example, examples_sorted
from search_utils.Query import Query
from search_utils.Sentence import Sentence, calc_sentiment_batch

logger = logging.getLogger(__name__)
debug = logger.debug


def expand_sentence(text_p: Union[str, Example],
                    word_indices: List[int],
                    query=None,
                    additional_mask_indices: List[int] = None,
                    schedule_idx=[-1]) -> List[Example]:
    if additional_mask_indices is None:
        additional_mask_indices = []

    if isinstance(text_p, Example):
        text = text_p.sentence
    else:
        text = text_p
    text = Sentence(text)
    word_indices = list(word_indices)
    word_indices = [wi for wi in word_indices if not all([s in ":,;.*" for s in text.words[wi]])]
    if len(word_indices) == 0:
        return []

    original_words = {i: text.words[i] for i in word_indices}
    max_words = query.consider_max_words if query is not None else Query(None).consider_max_words
    masked_sentence = Sentence(text.get_with_masked(word_indices + additional_mask_indices))
    predictions = masked_sentence.calc_mask_predictions(max_words)

    result = []
    for word_idx in word_indices:

        if not predictions[word_idx]:
            continue

        sentences = []
        for predicted_token, score in predictions[word_idx]:
            new_sen = text.replace_word(word_idx, predicted_token)
            sentences.append(new_sen)

        classification = calc_sentiment_batch(sentences)
        for i, (predicted_token, score) in enumerate(predictions[word_idx]):
            if original_words[word_idx] != predicted_token:
                if isinstance(text_p, str):
                    e = Example(sentences[i], classification[i], [(word_idx, score)],
                                pred_ind=[i],
                                sched_ind=[schedule_idx],
                                sent_ind=[0])
                else:
                    e = Example(sentences[i], classification[i],
                                text_p.changes + [(word_idx, score)],
                                pred_ind=text_p.prediction_indices + [i],
                                sched_ind=text_p.schedule_indices + [schedule_idx],
                                sent_ind=text_p.sentence_indices + [0])
                result.append(e)

    return result


def expand_sentence_neighbour(text_p: Union[str, Example], query, word_indices: List[int], schedule_idx) -> List[Example]:
    """

    Parameters
    ----------
    schedule_idx
    text_p
    word_indices
    query : search_helper.classes.Query.Query

    Returns
    -------

    """
    if isinstance(text_p, Example):
        text = text_p.sentence
    else:
        text = text_p
    text = Sentence(text)
    word_indices = list(word_indices)
    word_indices = [wi for wi in word_indices if not all([s in ":,;.*" for s in text.words[wi]])]

    if len(word_indices) == 0:
        return []

    word_indices = word_indices[:MBS_DEPTH]
    masked_text = Sentence(text.get_with_masked(word_indices))

    initial_example = [Example(masked_text.text, [], [], [], [], [])]
    results = []
    for word_idx in word_indices:
        debug(f"expand neighbours: {word_idx} of {word_indices}")
        tmp_results = examples_sorted(results, query.wanted_cls, query.c)[:MBS_BEAMS]
        results = []
        for interm_example in (initial_example if word_idx == word_indices[0] else tmp_results):
            intermediate_sen = Sentence(interm_example)
            predictions = Sentence(interm_example).calc_mask_predictions(query.consider_max_words)
            if not predictions[word_idx]:
                continue

            sentences = []
            for predicted_token, score in predictions[word_idx]:
                new_sen = intermediate_sen.replace_mask(word_idx, predicted_token)
                sentences.append(new_sen)

            classification = calc_sentiment_batch(sentences)
            for i, (predicted_token, score) in enumerate(predictions[word_idx]):
                results.append(Example(sentences[i], classification[i],
                                       interm_example.changes + [(word_idx, score)],
                                       pred_ind=interm_example.prediction_indices + [i],
                                       sched_ind=interm_example.schedule_indices + [schedule_idx],
                                       sent_ind=[0]))

    return examples_sorted(results, query.wanted_cls, query.c)

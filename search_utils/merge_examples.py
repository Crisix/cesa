import logging
from collections import defaultdict
from typing import List

from config import MIN_DISTANCE_BETWEEN_CHANGED_WORDS, MAX_EXAMPLES_TO_CONSIDER_FOR_MERGING
from search_utils.Example import Example, examples_sorted
from search_utils.Sentence import Sentence, calc_sentiment_batch

logger = logging.getLogger(__name__)
debug = logger.debug


def min_distance(c1: List[int], c2: List[int]):
    cmin = float("inf")
    for x in c1:
        for y in c2:
            cmin = min(cmin, abs(x - y))
    return cmin


def generate_merged_examples(original, examples: List[Example], query, curr_iter_idx):
    """

    Parameters
    ----------
    original : Sentence
    examples : List of Example
    query: search_helper.classes.Query.Query
    curr_iter_idx: int

    Returns
    -------
    list of merged examples
    """
    generated = []
    cs = query.c if isinstance(query.c, list) else [query.c]
    for c in cs:
        grouped_by_changed_indices = defaultdict(list)
        examples = examples_sorted(examples, query.wanted_cls, c)[:MAX_EXAMPLES_TO_CONSIDER_FOR_MERGING]
        for e in examples:
            if len(grouped_by_changed_indices[tuple(e.changed_word_indices())]) <= 5:
                grouped_by_changed_indices[tuple(e.changed_word_indices())].append(e)

        grouped_items = grouped_by_changed_indices.items()
        grouped_keys = [set(x) for x in grouped_by_changed_indices.keys()]
        if len(grouped_items) <= 1:  # -> nothing to merge
            return []

        if curr_iter_idx == -1:
            return [merge_all_changes_into_one(grouped_items, original)]

        debug(f"MERGING ({len(grouped_keys)}) {grouped_keys}")

        for a_idx, (changes_a, la) in enumerate(grouped_items):
            for b_idx, (changes_b, lb) in enumerate(grouped_items):

                # changes already done or changes overlap or changes too close together
                if set(changes_a + changes_b) in grouped_keys \
                        or len(set(changes_a).intersection(set(changes_b))) > 0 \
                        or min_distance(changes_a, changes_b) < MIN_DISTANCE_BETWEEN_CHANGED_WORDS:
                    continue

                resulting_edit_size = len(set(changes_a + changes_b))
                take_n = max(1, 5 - resulting_edit_size)

                # Merge examples
                a: Example
                b: Example
                for a in la[:take_n]:
                    for b in lb[:take_n]:
                        sa, sb = Sentence(a), Sentence(b)
                        new = list(original.words)
                        for na in a.changed_word_indices():
                            new[na] = sa.words[na]
                        for na in b.changed_word_indices():
                            new[na] = sb.words[na]

                        generated.append((" ".join(new),
                                          tuple(a.changes + b.changes),
                                          a.prediction_indices + b.prediction_indices,
                                          a.schedule_indices + b.schedule_indices,
                                          a.sentence_indices + a.sentence_indices
                                          ))
                        grouped_keys.append(set(dict(a.changes + b.changes).keys()))  # can indent 2?

    debug(f"MERGE: generated {len(generated)} MERGED (#c={len(cs)}) examples")

    if len(generated) == 0:
        return []
    unzipped = list(zip(*generated))
    sentence_list = list(unzipped[0])
    if len(sentence_list) == 0:
        return []
    sentiment_batch = calc_sentiment_batch(sentence_list)
    result = []
    for i, (sen, changes, pred_idx, sched_idx, sent_idx) in enumerate(generated):
        example = Example(sen, sentiment_batch[i], list(changes),
                          pred_ind=pred_idx,
                          sched_ind=sched_idx,
                          sent_ind=sent_idx)
        example.schedule_indices = sched_idx
        result.append(example)
    return result


def merge_all_changes_into_one(grouped_items, original) -> Example:
    # Merge all available -> check
    merge_all_changes_into_this_one = list(original.words)
    merge_changes = []
    pred_indices = []
    sched_indices = []
    sent_indices = []
    for changes_a, la in grouped_items:
        if len(changes_a) == 1 and len(la) > 0:
            e: Example = la[0]
            s = Sentence(e)
            merge_all_changes_into_this_one[changes_a[0]] = s.words[changes_a[0]]
            merge_changes += e.changes
            pred_indices.append(e.prediction_indices)
            sched_indices.append(e.schedule_indices)
            sent_indices.append(e.sentence_indices)
    merge_all_changes_into_this_one = " ".join(merge_all_changes_into_this_one)
    all_sentiment = calc_sentiment_batch([merge_all_changes_into_this_one])
    return Example(merge_all_changes_into_this_one, all_sentiment[0], merge_changes,
                   pred_indices, sched_indices, sent_indices)

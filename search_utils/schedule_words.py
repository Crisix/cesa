from enum import Enum
from typing import List, Tuple

import numpy as np

MINIMUM_ALLOWED_DISTANCE = 10
SPREAD_EVEN = True


class Edit(Enum):
    NORMAL, NEIGHBOURS = range(2)


def min_dist(val: int, lst: List):
    cmin = float("inf")
    for el in lst:
        cmin = min(cmin, abs(val - el))
    return cmin


def calc_combine_neighbors(word_indices: List[int]) -> (List[List[int]], List[int]):
    indices_sorted = np.array(sorted(word_indices))
    schedule = []
    cur = []
    last = float("-inf")
    for i in range(len(indices_sorted)):
        if abs(last - indices_sorted[i]) == 1:
            cur.extend([last, indices_sorted[i]])
        else:
            if len(cur) > 0:
                schedule.append(list(set(cur)))
            cur = []
        last = indices_sorted[i]
    if len(cur) > 0:
        schedule.append(list(set(cur)))
    # sort based on previous sorting
    for i in range(len(schedule)):
        slot = sorted([(v, word_indices.index(v)) for v in schedule[i]], key=lambda m: m[1])
        schedule[i] = list(list(zip(*slot))[0])
    # remove used indices
    remaining_word_indices = list(word_indices)
    for slot in schedule:
        for s in slot:
            remaining_word_indices.remove(s)
    return schedule, remaining_word_indices


def distribute_over_schedule(remaining_word_indices):
    schedule = []
    for t in remaining_word_indices:
        slot_distances = np.array([min_dist(t, slot) for (edit, slot) in schedule])
        slot_viable = slot_distances >= MINIMUM_ALLOWED_DISTANCE
        slot_len = np.array([len(slot) for slot in schedule])

        inserted = False
        visit_order = np.argsort(slot_len) if SPREAD_EVEN else range(len(schedule))
        for slot_idx in visit_order:
            if slot_viable[slot_idx] and not inserted:
                inserted = True
                schedule[slot_idx][1].append(t)
        if not inserted:
            schedule.append((Edit.NORMAL, [t]))
    return schedule


def merge_based_on_order(schedule_a, schedule_b, original_order):
    total = schedule_a + schedule_b
    order = [min([original_order.index(i) for i in x]) for (e, x) in total]
    return [x for _, x in sorted(zip(order, total))]  # sorts total based on order


def add_additional_masks(remaining_word_indices, ordered_word_indices):
    result = []
    for t in remaining_word_indices:
        additional_masks = []
        for o in ordered_word_indices:
            if abs(t - o) >= MINIMUM_ALLOWED_DISTANCE:
                additional_masks.append(o)
        result.append((Edit.NORMAL, ([t], additional_masks)))
    return result


def generate_schedule_multiple(word_gradients, top_k, multiple_words_at_once, combine_neighbours) -> List[Tuple[Edit, List[int]]]:
    """ old concept, bad idea, not used anymore """
    all_ordered_word_indices = np.argsort(word_gradients)[::-1].tolist()
    ordered_word_indices = all_ordered_word_indices[: top_k]
    if not multiple_words_at_once and not combine_neighbours:
        return [(Edit.NORMAL, [x]) for x in ordered_word_indices]

    if combine_neighbours:
        schedule, remaining_word_indices = calc_combine_neighbors(all_ordered_word_indices[:top_k + 5])
        schedule = [(Edit.NEIGHBOURS, slot) for slot in schedule]
    else:
        schedule, remaining_word_indices = [], ordered_word_indices

    if multiple_words_at_once:
        schedule = merge_based_on_order(schedule_a=schedule,
                                        schedule_b=distribute_over_schedule(remaining_word_indices),
                                        original_order=ordered_word_indices)
    else:
        [schedule.append((Edit.NORMAL, [x])) for x in remaining_word_indices]

    return schedule


def generate_schedule(word_gradients, top_k, mask_additional_words, mini_beam_search) -> List[Tuple[Edit, List[int]]]:
    schedule = []
    ordered_word_indices = np.argsort(word_gradients)[::-1][:top_k].tolist()

    if not mask_additional_words and not mini_beam_search:
        return [(Edit.NORMAL, ([x], [])) for x in ordered_word_indices]

    if mini_beam_search:
        schedule, _ = calc_combine_neighbors(ordered_word_indices)
        schedule = [(Edit.NEIGHBOURS, slot) for slot in schedule]

    if mask_additional_words:
        schedule.extend(add_additional_masks(ordered_word_indices, ordered_word_indices))
    else:
        [schedule.append((Edit.NORMAL, ([x], []))) for x in ordered_word_indices]

    return schedule

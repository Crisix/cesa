from typing import List

import numpy as np

from search_utils.Example import Example, examples_sorted
from search_utils.Result import Result


def any_in(list_a, list_b):
    return any([(x in list_b) for x in list_a])


def is_sufficient(e: Example, query):
    """

    Parameters
    ----------
    e : Example
    query : search_helper.classes.Query.Query

    Returns
    -------

    """
    idx = np.argmax(query.wanted_cls)
    return abs(query.wanted_cls[idx] - e.cls[idx]) < query.max_delta


def examples_are_sufficient(examples: List[Example], query) -> Result:
    """

    Parameters
    ----------
    examples: List of Example
    query : search_helper.classes.Query.Query

    Returns
    -------

    """
    c = query.c if isinstance(query.c, float) else query.c[0]
    already_used_indices = []
    result_by_change_idx = []
    rest = []

    viable = examples_sorted([e for e in examples if is_sufficient(e, query)], query.wanted_cls, c)
    for i in range(query.num_needed):
        if len(viable) > 0:
            wi = viable[0].changed_word_indices()
            already_used_indices.extend(wi)
            same = [v for v in viable if v.changed_word_indices() == wi]
            viable = [v for v in viable if v.changed_word_indices() != wi]

            # --------------------
            viable_index_used = [v for v in viable if any_in(v.changed_word_indices(), already_used_indices)]
            viable_index_unused = [v for v in viable if not any_in(v.changed_word_indices(), already_used_indices)]
            rest.extend(viable_index_used)
            viable = viable_index_unused
            # --------------------

            result_by_change_idx.append(same)
        else:
            result = Result(success=False, rest=viable + rest, examples=result_by_change_idx)
            result.query = query
            return result

    result = Result(success=True, rest=examples_sorted(viable + rest, query.wanted_cls, c), examples=result_by_change_idx)
    result.query = query
    return result

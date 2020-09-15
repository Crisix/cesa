from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch.nn.functional import mse_loss

from config import COST_PER_ADDITIONAL_WORD, WordIdx
from search_utils.Sentence import Sentence


def examples_sorted(examples, wanted_cls, c):
    return sorted(examples, key=lambda e: e.cf_loss(wanted_cls, c))


@dataclass
class Example:

    def __init__(self, sentence: str, classification, changes, pred_ind, sched_ind, sent_ind):
        self.cls = classification
        self.sentence = sentence
        self.changes: List[Tuple[WordIdx, float]] = changes
        self.prediction_indices = pred_ind  # n-th word in list of alternative words
        self.schedule_indices = sched_ind  # 1st highest gradient word, 2nd hgw, ...
        self.sentence_indices = sent_ind  # 1st sentence,... only != 0 if splitting text
        self.perplexity = None
        for (i, _) in changes:
            assert i < len(Sentence(self.sentence).words)
        assert len(self.schedule_indices) == len(self.prediction_indices) == len(self.changes)

    def changed_word_indices(self):
        return list(dict(self.changes).keys())

    def changed_word_distances(self):
        return list(dict(self.changes).values())

    def marked(self):
        ps = Sentence(self.sentence).words
        bold_indices = list(zip(*self.changes))[0]
        for i, wi in enumerate(bold_indices):
            ps[wi] = f"#{i + 1}#={ps[wi]}"
        return " ".join(ps)

    def used_mini_beam_search(self):
        cwi = self.changed_word_indices()
        min_dist = float("inf")
        for a in cwi:
            for b in cwi:
                if a != b:
                    min_dist = min(abs(a - b), min_dist)
        return min_dist == 1

    def cf_loss(self, y_prime, c):
        mse = mse_loss(torch.tensor(self.cls), torch.tensor(y_prime, dtype=torch.float32))
        d = list(dict(self.changes).values())
        theta = sum([d_i ** 2 for d_i in d]) + COST_PER_ADDITIONAL_WORD * len(d)
        return mse + c * theta

    def calc_perplexity(self):
        if self.perplexity is None:
            self.perplexity = Sentence(self.sentence).calc_perplexity()
        return self.perplexity

    def info(self):
        return f"pp={self.calc_perplexity():.2f}, " + self.__repr__()

    def __repr__(self):
        # DEBUG HELP
        left_right_window = 6
        sen = Sentence(self.sentence)
        changed_indices = self.changed_word_indices()
        sp = [w if i not in changed_indices else f"#{w}#" for i, w in enumerate(sen.words)]
        relevant_parts = ' '.join(sp)
        if len(relevant_parts) > 160:
            relevant_parts = []
            for i, (w_id, score) in enumerate(self.changes):
                if i != 0:
                    relevant_parts.append("[...]")
                word = sp[w_id]
                left = sp[max(0, w_id - left_right_window):w_id]
                right = sp[w_id + 1:w_id + left_right_window]
                relevant_parts.extend(left + [word] + right)
            relevant_parts = ' '.join(relevant_parts)

        return f"PRED_IDX={self.prediction_indices}, " \
               f"cls={np.round(self.cls, 2)}, " \
               f"idxs={changed_indices} " \
               f"dist={np.round(list(dict(self.changes).values()), 2)}, {relevant_parts}"

    def to_df(self, wanted_pos, c):
        return {
            "changes": self.changes,
            "positivity": self.cls,
            "cf_loss": self.cf_loss(wanted_pos, c),
            "num_changes": len(self.changes),
            "changed_indices": list(dict(self.changes).keys()),
            "sentence": self.marked(),
            "repr": self.__repr__()
        }

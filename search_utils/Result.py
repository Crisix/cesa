from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from config import WordIdx
from search_utils.Example import Example
from search_utils.Sentence import Sentence
from search_utils.Statistics import Statistics


@dataclass
class Result:
    success: bool = False
    rest: List[Example] = None
    examples: List[List[Example]] = None
    stats: Statistics = None
    sentence_map: List[Tuple[WordIdx, WordIdx]] = None
    query = None

    def simple_results(self):
        return [same_idx_list[0] for same_idx_list in self.examples]

    def total_valid_examples(self):
        return sum([len(sl) for sl in self.examples]) + len(self.rest)

    def info(self):
        result_str = ""
        sen = Sentence(self.stats.original_sentence)
        result_str += f"pp={sen.calc_perplexity():.2f}, {len(sen.words)} words, y={np.round(self.stats.original_classification, 2)}\n"
        result_str += f"Duration: {self.stats.total_duration} | {self.stats.find_matching_words_duration} searching words | {self.stats.merging_duration} merging.\n"
        result_str += f"{self.total_valid_examples()} examples, {len(self.rest)} in rest, found {len(self.examples)} of {self.query.num_needed} groups with different indices.\n"
        for e in self.simple_results():
            result_str += "\t" + e.info() + "\n"
        result_str += "Successful!\n" if self.success else "Query not fullfilled!"
        return result_str

    def sufficient(self):
        return len(self.examples) >= self.query.num_needed

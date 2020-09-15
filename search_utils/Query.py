from dataclasses import dataclass
from typing import Union, List


@dataclass
class Query:
    # Parameters
    wanted_cls: List[float]  # [0,1] for positive
    max_delta: float = 0.4  # allowed delta on wanted_classification
    c: Union[List[float], float] = 0.5
    num_needed: int = 1  # Number of examples, where different indices are changed

    # Algorithm choice
    mask_additional_words: bool = False
    mini_beam_search: bool = False
    allow_splitting: bool = False

    # Time-Quality-Tradeoff
    consider_top_k: int = 10  # do not change!
    consider_max_words: int = 250  # do not change!
    consider_max_sentences: int = 7  # do not change!

    def alg(self):
        return {(True, True): "MBS+ST",
                (True, False): "MBS",
                (False, True): "ST",
                (False, False): "BASE"}[(self.mini_beam_search, self.allow_splitting)]

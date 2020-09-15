import pandas as pd

from config import model_config
from generate_counterfactuals import generate_counterfactuals
from search_utils.Query import Query
from search_utils.Sentence import Sentence

model_config.load("imdb", evalution_model="gpt2")

num = 5
result = []
for wanted_positivity in range(num + 1):
    wanted_positivity = wanted_positivity / num
    wanted_cls = [(1 - wanted_positivity), wanted_positivity]
    max_delta = 50. / num / 100.
    print(f"{wanted_cls[1]}+-{max_delta}")
    # relative high consider_max_words becasue max_delta is small.
    # sent = "A decent story with some thrilling action scenes."
    # sent = "the year's best and most unpredictable comedy."
    sent = "an extremely unpleasant film."
    r = generate_counterfactuals(sent,
                                 Query(wanted_cls=wanted_cls, max_delta=max_delta))
    print(r.examples[0][0] if len(r.examples) > 0 else "----")
    result.append({
        "y'": f"{wanted_cls[1]:.1f} pm {max_delta:.1f}",
        "y": f"{r.examples[0][0].cls[1]:.2f}",
        "Counterfactual Example x' ": r.examples[0][0].sentence
    })

print("######")
print(f"Original cls {Sentence(sent).calc_sentiment()[1]}")
with pd.option_context("max_colwidth", 1000):
    print(pd.DataFrame(result).to_latex(index=False))

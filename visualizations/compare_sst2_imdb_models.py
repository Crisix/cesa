from config import model_config
from generate_counterfactuals import generate_counterfactuals
from search_utils.Query import Query
import pandas as pd

POSITIVE = [0, 1]
NEGATIVE = [1, 0]

data = [
    ("it 's a charming and often affecting journey .", NEGATIVE),
    ("although laced with humor and a few fanciful touches , the film is a refreshingly serious look at young women .", NEGATIVE),
    ("... the film suffers from a lack of humor ( something needed to balance out the violence ) ...", POSITIVE),
    ("in its best moments , resembles a bad high school production of grease , without benefit of song .", NEGATIVE)
]

all_result_str = ""
for idx, (sen, y_prime) in enumerate(data):
    results = []
    for ds in ["imdb", "sst-2"]:
        model_config.load(ds)
        r = generate_counterfactuals(sen, Query(y_prime, c=0.2))
        results.append({
            "Datensatz": ds,
            "Text": r.examples[0][0].sentence if len(r.examples) > 0 else "NO CF FOUND"
        })
    results.append({
        "Datensatz": "Original",
        "Text": sen
    })
    with pd.option_context("max_colwidth", 100000):
        all_result_str += pd.DataFrame(results).to_latex(index=False)

all_result_str = all_result_str \
    .replace("tabular", "tabularx") \
    .replace("\\begin{tabularx}", "\\begin{tabularx}{\\textwidth}")
print(all_result_str)
print(all_result_str)

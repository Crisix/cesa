import logging
import pickle
from datetime import date
from typing import List, Tuple

import nlp
import tqdm

from config import model_config, gpu_name
from generate_counterfactuals import generate_counterfactuals
from search_utils.Query import Query
from search_utils.Result import Result

logger = logging.getLogger(__name__)
info = logger.info

model_config.load("imdb", None)

imdb = nlp.load_dataset("imdb")
imdb_train, imdb_test = imdb["train"], imdb["test"]
dataset = imdb_test.shuffle(seed=42)  # otherwise labels are sorted

try:
    # noinspection PyUnresolvedReferences
    from google.colab import drive

    drive.mount('/content/drive')
except ModuleNotFoundError:
    info("probably not running on colab")

results: List[Tuple[int, Result]] = []

new_start = 0
for enm in tqdm.tqdm(range(new_start, len(dataset))):

    data = dataset[enm]
    x, y = data["text"], data["label"]

    x = model_config.tokenizer.clean_up_tokenization(x)
    if y == -1:
        info("y is -1 \ny is -1 \ny is -1 \ny is -1 \n")
        continue  # test data for SST-2 has label -1 (placeholder?)
    y_prime = 1 - y
    y_prime = [1 - y_prime, y_prime]

    for mbs in [True, False]:
        for allow_splitting in [True, False]:
            query = Query(wanted_cls=y_prime, max_delta=0.4, c=0.2, num_needed=1,
                          mask_additional_words=False,
                          mini_beam_search=mbs, allow_splitting=allow_splitting,
                          consider_top_k=20, consider_max_words=500, consider_max_sentences=8)
            r = generate_counterfactuals(x, query)
            results.append((enm, r))

    fname = f"{new_start}_to_{enm}_on_{gpu_name()}_imdb_{date.today()}.pickle"
    if enm % 5 == 0 and enm != 0:
        try:
            path = F"/content/drive/My Drive/{fname}"
            with open(path, "wb") as file:
                pickle.dump({"imdb": results}, file)
            info("saved")
        except Exception as e:
            info(f"probably not running on colab: {e}")
            # with open(fname, "wb") as file:
            #     pickle.dump({DATASET: results}, file)
            # info("saved")

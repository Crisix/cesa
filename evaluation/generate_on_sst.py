import logging
import pickle
from datetime import date
from typing import List, Tuple

import nlp
from tqdm import tqdm

from config import model_config, gpu_name
from generate_counterfactuals import generate_counterfactuals
from search_utils.Query import Query
from search_utils.Result import Result

logging.getLogger().setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
info = logger.info

model_config.load("sst-2", None)

sst2 = nlp.load_dataset('glue', 'sst2')
sst2_train, sst2_validation, sst2_test = sst2["train"], sst2["validation"], sst2["test"]
dataset = sst2_validation

try:
    # noinspection PyUnresolvedReferences
    from google.colab import drive

    drive.mount('/content/drive')
except ModuleNotFoundError:
    logger.warning("probably not running on colab?")

logging.getLogger().setLevel(logging.INFO)

results: List[Tuple[int, Result]] = []

new_start = 1 + 660
for enm in tqdm(range(new_start, len(dataset))):

    data = dataset[enm]
    x, y = data["sentence"], data["label"]

    x = model_config.tokenizer.clean_up_tokenization(x)
    assert y != -1
    y_prime = [y, 1 - y]

    for mbs in [True, False]:
        for st in [True, False]:
            query = Query(wanted_cls=y_prime, max_delta=0.4, c=0.2, num_needed=3,
                          mask_additional_words=False,
                          mini_beam_search=mbs, allow_splitting=st)
            r = generate_counterfactuals(x, query)
            results.append((enm, r))

    fname = f"{new_start}_to_{enm}_on_{gpu_name()}_sst-2_{date.today()}.pickle"
    if enm % 10 == 0:
        try:
            path = F"/content/drive/My Drive/{fname}"
            with open(path, "wb") as file:
                pickle.dump({"SST-2": results}, file)
            info("saved")
        except Exception as e:
            info(f"probably not running on colab: {e}")
            # with open(fname, "wb") as file:
            #     pickle.dump({DATASET: results}, file)
            # info("saved")

import pickle
import json
import matplotlib
from tqdm import tqdm

from generate_counterfactuals import calc_sentence_edit_schedule
from search_utils.Example import Example
from search_utils.Result import Result
from config import model_config
import numpy as np

from search_utils.Sentence import Sentence
from search_utils.split_sentence import get_sentence_word_mapping
from visualizations.highlight_gradients import text_color


def clean(s):
    return model_config.tokenizer.clean_up_tokenization(' '.join(Sentence(s).words)).replace(" - ", "-").lower()


def twofivefive(tpl):
    return tuple([int(t * 255) for t in tpl])


# https://stackoverflow.com/a/57915246/3991393
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Example):
            ex_dict = dict(obj.__dict__)
            ex_dict["sentence"] = Sentence(clean(obj.sentence)).words
            ex_dict["perplexity"] = Sentence(obj.sentence).calc_perplexity()
            del ex_dict["prediction_indices"]
            del ex_dict["schedule_indices"]
            del ex_dict["sentence_indices"]
            del ex_dict["changes"]
            return ex_dict
        else:
            return super(CustomEncoder, self).default(obj)


max_ds_idx = 0


def extract_colors(r, per_sentence):
    text = r.stats.original_sentence
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "blue"])
    text_s = Sentence(text)
    if not per_sentence:
        word_gradients = text_s.calc_gradients(r.query.wanted_cls)
        wgn = np.interp(word_gradients, (np.min(word_gradients), np.max(word_gradients)), (0., 1.))
        fg, bg = [], []
        for ind in range(len(wgn)):
            ctpl = cmap(wgn[ind])[:3]
            tc = twofivefive(text_color(ctpl))
            ctpl = twofivefive(ctpl)
            fg.append(str(tc)[1:-1])
            bg.append(str(ctpl)[1:-1])
        return fg, bg
    else:
        sw_map = get_sentence_word_mapping(text)
        edit_sentence_order = calc_sentence_edit_schedule(r.query, sw_map, text_s)
        fg, bg = [], []
        for enm_si, si in enumerate(edit_sentence_order):
            start, stop = sw_map[si]
            sub = model_config.tokenizer.clean_up_tokenization(" ".join(text_s.words[start:stop + 1]))
            subtext = Sentence(sub)
            word_gradients = np.array(subtext.calc_gradients(r.query.wanted_cls))
            word_gradients /= np.linalg.norm(word_gradients)
            wgn = np.interp(word_gradients, (np.min(word_gradients), np.max(word_gradients)), (0., 1.))
            for ind in range(len(wgn)):
                ctpl = cmap(wgn[ind])[:3]
                tc = twofivefive(text_color(ctpl))
                ctpl = twofivefive(ctpl)
                fg.append(str(tc)[1:-1])
                bg.append(str(ctpl)[1:-1])
        return fg, bg


json_all = []
last_ds_idx = -1

# root = "../evaluation/"
root = ""
sst_files = [f'{root}data/sst2/0_to_220_on_Tesla K80_sst-2.pickle',
             f'{root}data/sst2/221_to_660_on_Tesla P100-PCIE-16GB_sst-2.pickle',
             f'{root}data/sst2/661_to_870_on_Tesla T4_sst-2.pickle',
             f'{root}data/sst2/871_to_871_on_Tesla T4_sst-2.pickle']
imdb_files = [f'{root}data/imdb/0_to_100_on_Tesla T4_H2H2_imdb.pickle',
              f'{root}data/imdb/101_to_560_on_Tesla P100-PCIE-16GB_H2H2_imdb.pickle',
              f'{root}data/imdb/561_to_855_on_Tesla P100-PCIE-16GB_H2H2_imdb.pickle']

ag_news_files = [f'{root}data/ag_news/0_to_245_on_Tesla P100-PCIE-16GB_ag_news.pickle']

for files in tqdm([ag_news_files]):
    for generate_for_this in tqdm(files):
        with open(generate_for_this, "rb") as file:
            dataset, data = list(pickle.load(file).items())[0]
            model_config.load(dataset, "gpt2")
            json_datapoint = None
            result: Result
            for another_idx, (ds_idx, result) in enumerate(tqdm(data)):
                max_ds_idx = max(ds_idx, max_ds_idx)
                if last_ds_idx != ds_idx:
                    if another_idx != 0:
                        json_all.append(json_datapoint)
                    last_ds_idx = ds_idx
                    fg, bg = extract_colors(result, per_sentence=False)
                    fg_ps, bg_ps = extract_colors(result, per_sentence=True)

                    json_datapoint = {
                        'sentence': Sentence(clean(result.stats.original_sentence)).words,
                        'foreground': fg,
                        'foreground_per_sen': fg_ps,
                        'background': bg,
                        'background_per_sen': bg_ps,
                        'original_cls': result.stats.original_classification,
                        'wanted_cls': result.query.wanted_cls,
                        'original_ppl': np.round(Sentence(result.stats.original_sentence).calc_perplexity(), 2)
                    }

                json_datapoint[result.query.alg()] = {
                    'examples': [exl[:5] for exl in result.examples]
                }

    print(f"MAX: {max_ds_idx}")
    with open(f'/content/drive/My Drive/{dataset}_data.json', 'w', encoding='utf-8') as f:
        print("dumping")
        json.dump([json_elem for json_elem in json_all if json_elem is not None], f, ensure_ascii=False, cls=CustomEncoder)
        print("dumping done")

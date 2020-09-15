import pickle
import pyperclip

from config import model_config
from search_utils.Result import Result
from search_utils.Sentence import Sentence


def clean(s):
    return model_config.tokenizer.clean_up_tokenization(' '.join(Sentence(s).words)).replace(" - ", "-").lower()


def escape_latex(s):
    return s.replace("$", "\\$").replace('"', "''")


def tex_color(text, color, bold=True):
    if bold:
        text = "\\textbf{" + text + "}"
    return "\\textcolor{" + color + "}{" + text + "}"


imdb_files = ['../evaluation/data/imdb/0_to_100_on_Tesla T4_H2H2_imdb.pickle',
              '../evaluation/data/imdb/101_to_560_on_Tesla P100-PCIE-16GB_H2H2_imdb.pickle',
              '../evaluation/data/imdb/561_to_855_on_Tesla P100-PCIE-16GB_H2H2_imdb.pickle']

DATASET = "IMDb"
model_config.load(DATASET)

data = []
for subfile in imdb_files:
    with open(subfile, 'rb') as f:
        pickle_data = pickle.load(f)
        items = list(pickle_data.items())
        file_ds = items[0][0]
        print(f"reading {subfile}: {len(items[0][1])} items")
        assert file_ds.lower() == DATASET.lower()
        data.extend(items[0][1])


def example_as_latex_string(idx):
    example_with_ans = data[idx]
    res_obj: Result
    res_obj = example_with_ans[1]
    if len(res_obj.examples) == 0:
        return ""
    cf_ex = [escape_latex(w) for w in Sentence(res_obj.examples[0][0].sentence).words]
    origi = [escape_latex(w) for w in Sentence(res_obj.stats.original_sentence).words]
    assert len(cf_ex) == len(origi)
    i = 0
    result_str = f"""
\\begin{{figure}}[h]
\\begin{{center}}
\\begin{{tabular}}{{|l|c|c|}} 
\\multicolumn{{3}}{{c}}{{Variante: {res_obj.query.alg()}}} \\\\
\\hline
{{}} & Counterfactual Example & Original \\\\
\\hline
Perplexity & {res_obj.examples[0][0].calc_perplexity():.2f} & {Sentence(res_obj.stats.original_sentence).calc_perplexity():.2f} \\\\
Polarität & {res_obj.examples[0][0].cls[1]:.2f} & {res_obj.stats.original_classification[1]:.2f} \\\\
\\hline
\\end{{tabular}}
\\end{{center}}
\n
"""
    while i != len(cf_ex):
        if cf_ex[i] != origi[i]:
            combine_from, combine_to = i, i + 1
            while origi[combine_to] != cf_ex[combine_to]:
                combine_to += 1
            left_part = ' '.join(origi[combine_from:combine_to])
            right_part = ' '.join(cf_ex[combine_from:combine_to])
            change = "\\mbox{[" + tex_color(left_part, "blue", False) + " \\to{} " + tex_color(right_part, "blue", True) + "]} "
            result_str += change
            i += (combine_to - combine_from)
        else:
            result_str += cf_ex[i] + " "
            i += 1
    result_str += f"\n\\caption{{ Beispiel {idx // 4} aus der {DATASET} Evaluation ({res_obj.query.alg()}) }}\n\\label{{{idx // 4}_{res_obj.query.alg()}}}\n\\end{{figure}} "
    return result_str + "\n\n\n"


MBS_ST_OFFSET, MBS_OFFSET, ST_OFFSET, BASE_OFFSET = 0, 1, 2, 3

copy_me = ""
for ex_idx in [3 + 1,
               # 157 + 1,
               310 + 1,
               390 + 1,
               419 + 1,
               506 + 1,
               591 + 1,
               753 + 1,
               ]:
    copy_me += example_as_latex_string(4 * ex_idx + MBS_ST_OFFSET)
    copy_me += example_as_latex_string(4 * ex_idx + MBS_OFFSET)
    copy_me += example_as_latex_string(4 * ex_idx + ST_OFFSET)
    copy_me += example_as_latex_string(4 * ex_idx + BASE_OFFSET)

pyperclip.copy(copy_me)

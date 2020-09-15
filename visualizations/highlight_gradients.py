import re

import matplotlib
import pandas as pd

from generate_counterfactuals import calc_sentence_edit_schedule, generate_counterfactuals
from config import model_config, COST_PER_ADDITIONAL_WORD
from search_utils.Query import Query
from search_utils.Sentence import Sentence
from search_utils.split_sentence import get_sentence_word_mapping
import numpy as np

POSITIVE = [0, 1]
NEGATIVE = [1, 0]

# https://www.rogerebert.com/reviews/harry-potter-and-the-sorcerers-stone-2001
# ex, y_prime = """ During "Harry Potter and the Sorcerer's Stone," I was pretty sure I was watching a classic, one that will be around for a long time, and make many generations of fans. It takes the time to be good. It doesn't hammer the audience with easy thrills, but cares to tell a story, and to create its characters carefully. Like "The Wizard of Oz," "Willy Wonka and the Chocolate Factory," "Star Wars" and "E.T.," it isn't just a movie but a world with its own magical rules. And some excellent Quidditch players. """, NEGATIVE

# ex, y_prime = """Brilliant adaptation of the story of Bletchley Park and the cryptanalysis team, ran by Alan Turing, that cracked the code of the German Enigma Machine during World War II. Featuring an outstanding starring performance from Benedict Cumberbatch as war hero Turning and supporting acts from a brilliant cast including Keira Knightley, Charles Dance and Mark Strong, 'The Imitation Game' is a powerful and eminently well-made biopic that illuminates the facts whilst respecting the story it is based upon. The English-language debut of 'Headhunters' director Morten Tyldum, this British World War II thriller is a highly conventional story about humanity that creates a fascinating character, anchored by a hypnotically complex performance.""", NEGATIVE

ex, y_prime = 'To this day, this is still my favorite pixar film. The animation is stellar, its heartwarming, funny and proves that pixar movies are always bound to be great (except for cars 2 but thats a different story). This has a shot at the title "best movie of the century"', NEGATIVE


def latex_escape(escape_me):
    # https://stackoverflow.com/questions/16259923/how-can-i-escape-latex-special-characters-inside-django-templates
    conv = {'&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{',
            '}': r'\}', '~': r'\textasciitilde{}', '^': r'\^{}', '\\': r'\textbackslash{}',
            '<': r'\textless{}', '>': r'\textgreater{}', '"': "''"}
    regex = re.compile('|'.join(re.escape(str(key)) for key in sorted(conv.keys(), key=lambda item: - len(item))))
    return regex.sub(lambda match: conv[match.group()], escape_me)


def text_color(background):
    return (0., 0., 0.) if ((0.299 * background[0]) + (0.587 * background[1]) + (0.114 * background[2])) > 0.5 else (1., 1., 1.)


# noinspection PyUnresolvedReferences
def generate_gradient_highlights():
    text = Sentence(ex)
    word_gradients = np.array(text.calc_gradients(y_prime))
    word_gradients /= np.linalg.norm(word_gradients)
    wgn = np.interp(word_gradients, (np.min(word_gradients), np.max(word_gradients)), (0., 1.))

    """
    \\newcommand{\\reducedstrut}{\\vrule width 0pt height .9\\ht\\strutbox depth .9\\dp\\strutbox\\relax}
    \\newcommand{\\mycb}[3]{%
      \\begingroup
      \\setlength{\\fboxsep}{0pt}%  
      \\colorbox[rgb]{#1}{ \\strut \\textcolor[rgb]{#2}{#3} }%
      \\endgroup
    }
    """
    result = ""  # new command overwritten error

    for cmap in [
        # matplotlib.colors.LinearSegmentedColormap.from_list("", ["blue", "white", "red"]),
        # matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "forestgreen"]),
        # matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "orangered"]),
        # matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "crimson"]),
        matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "blue"]),
        # # matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "red"]),
        # # matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "black"]),
    ]:
        result += f""
        for ind, w in enumerate(text.words):
            ctpl = cmap(wgn[ind])[:3]
            tc = str(text_color(ctpl))[1:-1]
            ctpl = [round(v, 3) for v in ctpl]
            rgba = str(ctpl)[1:-1]
            result += f"\\mycb{{{rgba}}}{{{tc}}}{{{latex_escape(w)}}}\\allowbreak"
        result += f"\n\\\\ Top 10: {', '.join(np.array(text.words)[np.argsort(-wgn)][:10])}\\ \n\n\n"

        # Sentence-wise calc gradients
        sw_map = get_sentence_word_mapping(text.text)
        edit_sentence_order = calc_sentence_edit_schedule(Query(y_prime), sw_map, text)

        for enm_si, si in enumerate(edit_sentence_order):
            start, stop = sw_map[si]
            sub = model_config.tokenizer.clean_up_tokenization(" ".join(text.words[start:stop + 1]))

            subtext = Sentence(sub)
            word_gradients = np.array(subtext.calc_gradients(y_prime))
            word_gradients /= np.linalg.norm(word_gradients)
            wgn = np.interp(word_gradients, (np.min(word_gradients), np.max(word_gradients)), (0., 1.))

            result += f"{enm_si + 1} Satz (vorher {si + 1}. Satz): "
            for ind, w in enumerate(subtext.words):
                ctpl = cmap(wgn[ind])[:3]
                tc = str(text_color(ctpl))[1:-1]
                ctpl = [round(v, 3) for v in ctpl]
                rgba = str(ctpl)[1:-1]
                result += f"\\mycb{{{rgba}}}{{{tc}}}{{{latex_escape(w)}}}\\allowbreak"
            result += "\\\\ \n\n"

    return result


def generate_example_table():
    result = ""
    text = Sentence(ex)
    for query in [
        Query(wanted_cls=y_prime, c=0.2, num_needed=5, mini_beam_search=False, allow_splitting=False, consider_top_k=10),
        Query(wanted_cls=y_prime, c=0.2, num_needed=5, mini_beam_search=True, allow_splitting=False, consider_top_k=10),
        Query(wanted_cls=y_prime, c=0.2, num_needed=5, mini_beam_search=False, allow_splitting=True, consider_top_k=10),
        Query(wanted_cls=y_prime, c=0.2, num_needed=5, mini_beam_search=True, allow_splitting=True, consider_top_k=10),
    ]:
        q_result = generate_counterfactuals(ex, query)
        print(q_result)
        ex_df = []
        for change_group in q_result.examples:
            for e in change_group:
                se = Sentence(e.sentence)
                d = e.changed_word_distances()
                cwi = e.changed_word_indices()
                entry = {"Original": ', '.join([text.words[wi] for wi in cwi]),
                         "Counterfactual": f"{', '.join([se.words[wi] for wi in cwi])}",
                         "Klassifikation": f"{e.cls[1]:.2f}",
                         "Distanz": f"{sum([d_i ** 2 for d_i in d]) + COST_PER_ADDITIONAL_WORD * len(d):.2f}"
                         }
                ex_df.append(entry)
        ex_df = pd.DataFrame(ex_df)
        # result += f"\n\n\nOriginale Klassifikation: {text.calc_sentiment()[1]:.2f} \\\\ \n"
        # result += f"\nMBS={query.mini_beam_search}, ST={query.allow_splitting}, MAX\\_WORDS={query.consider_max_words} \\\\ \n"
        result += "\n\n"
        result += ex_df.to_latex(index=False, caption=f"{query.alg()} (Originale Klassifikation: {text.calc_sentiment()[1]:.2f})")

    return result


if __name__ == '__main__':
    model_config.load("imdb", None)
    result = generate_gradient_highlights()
    result += "\n"
    result += generate_example_table()
    try:
        import pyperclip

        pyperclip.copy(result)
    except ImportError as e:
        print(e)

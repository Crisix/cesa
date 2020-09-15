import matplotlib.gridspec as grid_spec
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm

from generate_counterfactuals import _gen_cf_ex
from config import model_config
from search_utils.Query import Query
from search_utils.Sentence import Sentence
from search_utils.expand_sentence import expand_sentence


def get_scatter_data(word_idx):
    examples = expand_sentence(text, [word_idx])
    return [(ex.changed_word_distances()[0], ex.cls[1]) for ex in examples]


model_config.load("imdb")

text = "the acting is fine but the script is about as interesting as a recording of conversations at the wal-mart checkout line ."

# NOOS
# NOOS
# text = "I really liked this movie."
# text = "it 's a bad thing when a movie has about as much substance as its end credits blooper reel ."
# text = "as it is a loose collection of not-so-funny gags, scattered moments of lazy humor"
# text = "if you enjoy more thoughtful comedies with interesting conflicted characters ; this one is for you ."
# text = "no place for this story to go but down"
# text = "minority report is exactly what the title indicates , a report ."
# text = "it 's refreshing to see a girl-power movie that does n't feel it has to prove anything ."
# text = "it has all the excitement of eating oatmeal ."
# text = "his healthy sense of satire is light and fun ..."
# text = "Ultimately feels empty and unsatisfying, like swallowing a Communion wafer without the wine."
# text = "the action sequences are fun and reminiscent of combat scenes from the star wars series ."
# text = "with jump cuts , fast editing and lots of pyrotechnics , yu clearly hopes to camouflage how bad his movie is ."
# text = "why make a documentary about these marginal historical figures ?"
# text = "the character of zigzag is not sufficiently developed to support a film constructed around him ."
# text = "watchable up until the point where the situations and the dialogue spin hopelessly out of control"

text = model_config.tokenizer.clean_up_tokenization(text)

s = Sentence(text)
result = _gen_cf_ex(text, Query(wanted_cls=[0., 1.], max_delta=0.4, num_needed=5, consider_max_words=500, consider_top_k=15))
print(result.info())
result = [lst[0] for lst in result.examples]
data = {i: get_scatter_data(i) for i in range(len(s.words))}

colors = ["red", "green", "orange", "magenta", "lawngreen"]
# cmap_scale = cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=len(result)), cmap=cm.gist_rainbow)

fig = plt.figure(figsize=(10, 14))
gs = grid_spec.GridSpec(nrows=len(s.words), ncols=2, wspace=0, hspace=0.0001, width_ratios=[0.1, 1], figure=fig)
txts = []
ax_lst = []
for i, w in enumerate(s.words):
    ax1 = fig.add_subplot(gs[i, 1])
    ax1.text(-0.2, 0.5, w, fontsize=14, fontweight="bold", va="center", ha="center")  # adding text to ax1
    ax_lst.append(ax1)
    # ax1.spines['bottom'].set_visible(False)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.2, 1.2)
    ax1.set_yticks([0, 1])
    if i % 2 == 0:
        ax1.yaxis.tick_right()
    if len(data[i]) != 0:
        ax1.scatter(*zip(*data[i]), alpha=0.3, s=22, c="black")
        for ci, e in enumerate(result):
            if i in e.changed_word_indices():
                dist = e.changed_word_distances()[e.changed_word_indices().index(i)]
                cls = e.cls[1]
                # col = cmap_scale.to_rgba(ci)
                col = colors[ci]
                ax1.scatter([dist], [cls], s=35, color=col)
                t = ax1.annotate(Sentence(e.sentence).words[i], (dist + 0.02, cls + 0.2), fontsize=18)
                t.set_bbox(dict(facecolor='white', alpha=0.65, edgecolor=None, boxstyle="round"))
                txts.append(t)
    else:
        ax1.axvspan(-0.05, 1.05, alpha=0.2, color='grey')
# gs.update(hspace=-0.5)
# adjust_text(txts)

fig.text(0.5, 0.04, 'Entfernung zum Original', ha='center')
fig.text(1-0.04, 0.5, "Polarität", va='center', rotation='vertical')
fig.tight_layout()
fig.savefig("saved_plots/multiscatter_newhp.pdf")

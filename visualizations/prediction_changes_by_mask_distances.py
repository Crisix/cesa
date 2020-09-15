import pickle
from builtins import list
from collections import defaultdict

import matplotlib.pyplot as plt
import nlp
import numpy as np
import seaborn as sns
from config import model_config
from search_utils.Sentence import Sentence

root = "/content/drive/My Drive/"
# root = ""

sst2 = nlp.load_dataset('glue', 'sst2')
sst2_train, sst2_validation, sst2_test = sst2["train"], sst2["validation"], sst2["test"]
dataset = sst2_validation


def find_differences(original, alternative):
    distance_ = 0.
    wdiff_ = 0.
    dist_diff_only_ = 0.
    for oword, oscore in original:
        if oword in dict(alternative).keys():
            distance_ += abs(oscore - dict(alternative)[oword])
            dist_diff_only_ += abs(oscore - dict(alternative)[oword])
        else:
            wdiff_ += 1
            distance_ += 1.
    return distance_ / len(original), wdiff_ / len(original), dist_diff_only_ / len(original)


def one_mask():
    max_words = 20
    result = defaultdict(list)
    result_dist_diff_only = defaultdict(list)
    result_wdiff = defaultdict(list)

    for enm in range(len(dataset)):

        print(f"\nSentence {enm}: ", end="")
        data = dataset[enm]
        x, y = data["sentence"], data["label"]
        x = model_config.tokenizer.clean_up_tokenization(x)

        y = [0, 1] if y == 1 else [1, 0]
        y_prime = [1 - y[0], 1 - y[1]]

        s = Sentence(x)

        word_gradients = s.calc_gradients(y_prime)
        sorted_highest = np.argsort(word_gradients)[::-1]

        for observed_idx in sorted_highest[:10]:
            # observed_idx = sorted_highest[0]
            print(f"{observed_idx},", end="")
            sdir = 1 if len(s.words) - observed_idx > observed_idx else -1

            alt_s = Sentence(s.get_with_masked([observed_idx]))
            original_answer = alt_s.calc_mask_predictions()[observed_idx]

            if len(original_answer) != 0:

                for mask_distance in range(1, max_words):
                    if observed_idx + mask_distance * sdir < 0 or observed_idx + mask_distance * sdir >= len(alt_s.words):
                        continue

                    new_sen = Sentence(alt_s.get_with_masked([observed_idx + mask_distance * sdir, observed_idx]))
                    alt_sen_pred = new_sen.calc_mask_predictions()[observed_idx]

                    avg_distance, avg_word_diff, dist_diff_only = find_differences(original_answer, alt_sen_pred)

                    # print(f"Mask offset {mask_distance}: dist={avg_distance:.3f}  word_dist={avg_word_diff:.3f}")
                    result[mask_distance].append(avg_distance)
                    result_wdiff[mask_distance].append(avg_word_diff)
                    result_dist_diff_only[mask_distance].append(dist_diff_only)

        if enm % 50 == 0 or enm == len(dataset) - 1:
            fig = plt.figure(figsize=(11, 8))
            plt.title("Relation Bewertung der Wörter zur Nähe des nächsten [MASK]-Token")
            plt.xlabel("Entfernung zum zusätzlichen [MASK]-Token")
            plt.xlim(0, max_words)
            plt.ylim(0., 0.65)
            plt.ylabel("Veränderung der Bewertung")

            idx, mean, std = list(zip(*[(md, np.mean(lst), np.std(lst)) for (md, lst) in result_wdiff.items()]))
            mean = np.array(mean)
            std = np.array(std)
            plt.plot(idx, mean, color='r', label="Wort-Unterschiede")
            plt.fill_between(idx, mean - std, mean + std, color='r', alpha=.2)

            idx, mean, std = list(zip(*[(md, np.mean(lst), np.std(lst)) for (md, lst) in result_dist_diff_only.items()]))
            mean = np.array(mean)
            std = np.array(std)
            plt.plot(idx, mean, color='green', label="Distanz-Unterschiede")
            plt.fill_between(idx, mean - std, mean + std, color='green', alpha=.2)

            plt.xticks(idx)
            plt.legend()
            plt.savefig(f'{root}saved_plots/all/_besser_{enm}.png')
            # plt.show()
            plt.close(fig)


def two_mask():
    max_words = 15
    result = defaultdict(list)
    result_dist_diff_only = defaultdict(list)
    result_wdiff = defaultdict(list)

    for enm in range(len(dataset)):

        print(f"\nSentence {enm}: ", end="")
        data = dataset[enm]
        x, y = data["sentence"], data["label"]
        x = model_config.tokenizer.clean_up_tokenization(x)

        y = [0, 1] if y == 1 else [1, 0]
        y_prime = [1 - y[0], 1 - y[1]]

        s = Sentence(x)

        word_gradients = s.calc_gradients(y_prime)
        sorted_highest = np.argsort(word_gradients)[::-1]

        for observed_idx in sorted_highest[:10]:
            print(f"{observed_idx},", end="")

            alt_s = Sentence(s.get_with_masked([observed_idx]))
            original_answer = alt_s.calc_mask_predictions()[observed_idx]

            if len(original_answer) != 0:
                for mask_distance1 in range(-max_words, max_words + 1):
                    for mask_distance2 in range(-max_words, max_words + 1):

                        if not (0 <= observed_idx + mask_distance1 < len(alt_s.words)):
                            continue
                        if not (0 <= observed_idx + mask_distance2 < len(alt_s.words)):
                            continue

                        new_sen = Sentence(alt_s.get_with_masked([observed_idx + mask_distance1, observed_idx]))
                        new_sen = Sentence(new_sen.get_with_masked([observed_idx + mask_distance2, observed_idx]))
                        alt_sen_pred = new_sen.calc_mask_predictions()[observed_idx]

                        avg_distance, avg_word_diff, dist_diff_only = find_differences(original_answer, alt_sen_pred)

                        result[(mask_distance1, mask_distance2)].append(avg_distance)
                        result_wdiff[(mask_distance1, mask_distance2)].append(avg_word_diff)
                        result_dist_diff_only[(mask_distance1, mask_distance2)].append(dist_diff_only)

        if enm % 2 == 0 or enm == len(dataset) - 1:

            all_variants = [(result, "result"), (result_wdiff, "wdiff"), (result_dist_diff_only, "ddiff")]

            with open('used_data.pickle', 'wb') as handle:
                pickle.dump(all_variants, handle)

            for res, name in all_variants:
                data = [(k, np.mean(v)) for k, v in res.items()]
                matrix = np.zeros(shape=(2 * max_words + 1, 2 * max_words + 1))
                for (i, j), m in data:
                    matrix[(i + max_words), j + max_words] = m

                plt.figure(figsize=(15, 12))
                ax = sns.heatmap(np.flip(matrix, axis=0), linewidth=0.0,
                                 xticklabels=list(range(-max_words, max_words + 1)),
                                 yticklabels=list(reversed(range(-max_words, max_words + 1))))

                ax.set_title("Durchschnittliche Veränderung der Wörter bei 2 MASK-Tokens")
                plt.savefig(f'{root}saved_plots/2d/{name}_{enm}.pdf')
                plt.close()




model_config.load("sst-2", None)
two_mask()

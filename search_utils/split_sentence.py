import logging
from typing import List, Tuple

import nlp
from nltk import tokenize as nltk_tokenizer

from config import model_config
from search_utils.Sentence import Sentence

logger = logging.getLogger(__name__)
debug = logger.debug

MIN_SEN_LEN = 5


def clean_for_comparison(t):
    t = model_config.tokenizer.clean_up_tokenization(t)
    t = t.replace("<br /><br />", "")
    t = t.replace("<br />", "")
    t = t.replace("<br>", "")
    t = t.replace("<br >", "")
    # for s in "!\"'§$%&/()[]{}\\|:,;=?":
    #     t = t.replace(s, " ")
    for i in range(10):
        t = t.replace(" ", "")
    t = t.strip()
    return t


def get_sentence_word_mapping(text: str) -> List[Tuple[int, int]]:
    # can be longer than original, because Sentence(..) limited to 512 tokens
    tok_sen = nltk_tokenizer.sent_tokenize(text)
    original = model_config.tokenizer.clean_up_tokenization(" ".join(Sentence(text.lower().strip()).words))
    sentence = Sentence(original)
    word_sentence_map = []
    last_start = 0
    for xp in tok_sen:
        cxp = clean_for_comparison(xp.lower())
        for i in range(last_start, len(sentence.words) + 1):
            ccomp = clean_for_comparison("".join(sentence.words[last_start:i]))
            if ccomp == cxp:
                word_sentence_map.append((last_start, i - 1))
                last_start = i
                break
    if last_start != len(sentence.words):
        word_sentence_map.append((last_start, len(sentence.words)))

    # merge small sentences (<6) into neighbors
    while True:
        dists = [(stop - start) for (start, stop) in word_sentence_map]
        if all([d > MIN_SEN_LEN for d in dists]):
            return word_sentence_map
        else:
            # if True not in [d > MIN_SEN_LEN for d in dists]:
            #     return word_sentence_map
            sen_idx = [d > MIN_SEN_LEN for d in dists].index(False)
            # calc left side sen len
            if sen_idx - 1 < 0:
                left_len = None
            else:
                o_start, o_stop = word_sentence_map[sen_idx - 1]
                left_len = o_stop - o_start

            # calc right side sen len
            if sen_idx + 1 >= len(word_sentence_map):
                right_len = None
            else:
                o_start, o_stop = word_sentence_map[sen_idx + 1]
                right_len = o_stop - o_start

            if right_len is None and left_len is None:
                return word_sentence_map
            elif left_len is None or (right_len is not None and left_len < right_len):  # merge with right
                new_entry = (word_sentence_map[sen_idx][0], word_sentence_map[sen_idx + 1][1])
                word_sentence_map[sen_idx:sen_idx + 2] = [new_entry]
            elif right_len is None or (right_len is not None and right_len <= left_len):  # merge with left
                new_entry = (word_sentence_map[sen_idx - 1][0], word_sentence_map[sen_idx][1])
                word_sentence_map[sen_idx - 1:sen_idx + 1] = [new_entry]


def __evaluate():
    result = []
    sst = nlp.load_dataset('glue', 'sst2')
    dataset = sst["test"]
    # imdb = nlp.load_dataset('imdb')
    # dataset = imdb["test"]
    for enm in range(len(dataset)):
        data = dataset[enm]
        # x, y = data["text"], data["label"]
        x, y = data["sentence"], data["label"]
        b = nltk_tokenizer.sent_tokenize(x)
        a = get_sentence_word_mapping(x)
        if len(a) > 1:
            result.append(a)

        print(f"{len(Sentence(x).input_ids)} {' ' * 10 if len(a) != len(b) else ''} {len(a)} == {len(b)}")
        # assert len(a) == len(b)
    print(result)

# model_config.load("sst-2")
# __evaluate()

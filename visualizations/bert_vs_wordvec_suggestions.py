import gensim.downloader as api
import pandas as pd
from gensim.models.keyedvectors import Word2VecKeyedVectors
from config import model_config
from search_utils.Sentence import Sentence

wv = api.load('word2vec-google-news-300')
wv: Word2VecKeyedVectors

s = """my thoughts were focused on the characters ."""
s = model_config.tokenizer.clean_up_tokenization(s)
sen_s = Sentence(s)

df_wv = dict()
df = dict()
for i in range(len(sen_s.words)):

    word = sen_s.words[i]
    if word in ".:,?!-(){}[]/\\|&%":
        continue

    new_s = sen_s.get_with_masked(i)
    bert_preds = Sentence(new_s).calc_mask_predictions()[i]

    wv_preds = wv.most_similar(word, topn=15, restrict_vocab=200_000)

    df_wv[word] = list(list(zip(*wv_preds))[0])
    df[word] = list(zip(*bert_preds))[0][:15]

df = pd.DataFrame(df)
df_wv = pd.DataFrame(df_wv)
print("BERT")
print(df.to_latex())
print("WordVec")
print(df_wv.to_latex())

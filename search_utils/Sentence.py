import gc
import logging
import math
import string
from typing import List, Tuple
from typing import Union

import numpy as np
import torch
from torch.nn.functional import mse_loss

from config import SCALING_LOW, SCALING_HIGH, SUB_BATCH_SIZE, WordIdx, TokenIdx
from config import device, model_config
from search_utils.Query import Query
from search_utils.schedule_words import generate_schedule

logger = logging.getLogger(__name__)

punctuations = ".,:;&()/*[]{}\\!\"§$%-_<>#+"


def contains_punctuation(word):
    if word == "[MASK]":  # catch special case for [ and ]
        return False
    for p in punctuations:
        if p in word:
            return True
    return False


def is_valid_token(t):
    if t in string.punctuation or t == '...' or t == '।':
        return False
    if t.startswith("##"):
        return False
    if t.startswith("[unused_punctuation"):
        return False
    return True


def replace_with_mask(text: str, start, stop):
    new_text = list(text)
    new_text[start:stop] = "[MASK]"
    return "".join(new_text)


def scale_relatedness(s):
    # 0 = good change, 1 = bad change
    return 0. if s >= SCALING_HIGH else (1. if s <= SCALING_LOW else 1. - ((s - SCALING_LOW) / (SCALING_HIGH - SCALING_LOW)))


class Sentence:

    def __init__(self, text):
        if not isinstance(text, str):
            text = text.sentence
        if model_config.MODELS_ARE_UNCASED:
            text = text.lower()
        text = text.replace("[mask]", "[MASK]").replace("[sep]", "[SEP]")
        text = text.replace("<br />", " ")  # for IMDB dataset
        self.text = text
        encoded_dict = model_config.tokenizer.encode_plus(text, max_length=512, truncation=True, return_offsets_mapping=True)
        self.offset_mapping = encoded_dict["offset_mapping"]
        self.input_ids = encoded_dict['input_ids']
        self.attention_mask = encoded_dict['attention_mask']
        self.word_tokens: List[Tuple[str, List[Tuple[int, int]]]] = self.__process_offset_mapping(self.offset_mapping)
        self.words = self.__extract_words()
        self.t2w = self.init_token_to_word()
        self.w2t = self.init_word_to_token()

    def init_token_to_word(self):
        t2w = []
        for w_idx, (word, token_list) in enumerate(self.word_tokens):
            t2w.extend([w_idx for _ in token_list])
        return t2w

    def init_word_to_token(self):
        w2t = dict()
        ti = 0
        for w_idx, (word, token_list) in enumerate(self.word_tokens):
            for _ in token_list:
                w2t[w_idx] = w2t.get(w_idx, []) + [ti]
                ti += 1
        return w2t

    def calc_gradients(self, wanted_classification):
        model_config.sentiment_model.zero_grad()
        input_ids = torch.tensor(self.input_ids).unsqueeze(0).to(device)
        attention_mask = torch.tensor(self.attention_mask).unsqueeze(0).to(device)
        inp_emb = model_config.sentiment_model.get_input_embeddings()(input_ids).clone().detach().to(device).requires_grad_(True)
        result = model_config.sentiment_model(inputs_embeds=inp_emb, attention_mask=attention_mask)
        y_pred_gpu = torch.softmax(result[0], dim=1)
        y_pred = y_pred_gpu.cpu()

        loss = mse_loss(y_pred, torch.tensor([wanted_classification], dtype=torch.float32))
        loss.backward()

        inp_emb_norm = torch.norm(inp_emb.grad[0], p=2, dim=1)
        token_gradients = inp_emb_norm.cpu().numpy()  # vec norm of each word embedding gradient

        word_gradients = []
        current = 1  # to ignore [CLS] token
        for word, token_list in self.word_tokens:
            current_word_grad = []
            for _ in token_list:
                current_word_grad.append(token_gradients[current])
                current += 1
            word_gradients.append(np.mean(current_word_grad))
        model_config.sentiment_model.zero_grad()
        del input_ids, attention_mask, inp_emb, result, y_pred, y_pred_gpu, loss, inp_emb_norm, token_gradients
        torch.cuda.empty_cache()
        assert len(word_gradients) == len(self.words)
        return word_gradients

    def calc_edit_schedule(self, query):
        """

        Parameters
        ----------
        query: search_helper.classes.Query.Query

        Returns
        -------
        List[Tuple[Edit, List[int]]]
        List of edit operation with corresponding word indices which should be edited.
        """
        word_gradients = self.calc_gradients(query.wanted_cls)
        return generate_schedule(word_gradients, query.consider_top_k, query.mask_additional_words, query.mini_beam_search)

    def calc_perplexity(self):
        # print("calculating perplexity, should not happen while running search")
        with torch.no_grad():
            token_ids = model_config.perplexity_tokenizer.encode(self.text, max_length=512, truncation=True)
            token_ids = torch.tensor(token_ids).unsqueeze(0).to(device)
            return math.exp(model_config.perplexity_model(token_ids, labels=token_ids)[0].detach().cpu().numpy())

    def calc_mask_predictions(self, max_num=None):
        if max_num is None:
            max_num = Query(None).consider_max_words

        indices = [i - 1 for i, x in enumerate(self.input_ids) if x == model_config.tokenizer.mask_token_id]
        assert len(indices) != 0, "cant use calc_mask_predictions for sentence without mask token, use calc_word_predictions"
        return self.calc_word_predictions(indices if len(indices) > 1 else indices[0], max_num)

    def calc_word_predictions(self, indices: Union[List[TokenIdx], TokenIdx], max_num):
        predictions = self._calc_word_predictions()

        result = dict()
        _indices = [indices] if isinstance(indices, int) else indices
        for mask_index in _indices:
            mask_index += 1  # [CLS] offset
            word_indices_sorted = np.argsort(-predictions[0, mask_index])
            subresult = []
            for w in range(max_num):
                predicted_token = model_config.tokenizer.convert_ids_to_tokens([(word_indices_sorted[w])])[0]
                relatedness = float(predictions[0, mask_index, word_indices_sorted[w]])
                if relatedness < SCALING_LOW:
                    break
                if is_valid_token(predicted_token):
                    scaled_m = scale_relatedness(relatedness)
                    # scaled_m = relatedness # Only for evaluatiing scaling function
                    subresult.append((predicted_token, scaled_m))
            key = self.t2w[mask_index - 1]

            if key not in result:
                result[key] = subresult
            else:
                result[f"{key}_{mask_index}"] = subresult

        return result

    def _calc_word_predictions(self):
        with torch.no_grad():
            input_ids = torch.tensor(self.input_ids).unsqueeze(0).to(device)
            attention_mask = torch.tensor(self.attention_mask).unsqueeze(0).to(device)
            result = model_config.model(input_ids, attention_mask=attention_mask)[0]
            result_cpu = result.to('cpu').numpy()
            del result, input_ids, attention_mask
            return result_cpu

    def __process_offset_mapping(self, offset_mapping):
        """ Needs to be done, because one doesnt know how many tokens a word has,
            thats before the masked word -> which id has the masked token? """
        # put tokens belonging to the same word in the same list
        last_stop, grouped_tokens, token_accumulator = 0, [], []
        for token_start, token_stop in offset_mapping[1:-1]:
            current = self.text[token_start:token_stop]
            if current == "[MASK]":
                grouped_tokens.append(token_accumulator)
                grouped_tokens.append([(token_start, token_stop)])
                token_accumulator = []
            else:
                # if token_start == last_stop and current not in ".:,;&(/)":
                connected_to_last = token_start == last_stop
                contains_punct = contains_punctuation(current)
                if connected_to_last and not contains_punct:
                    # Add to current word
                    token_accumulator.append((token_start, token_stop))
                elif contains_punct:
                    grouped_tokens.append(token_accumulator)
                    grouped_tokens.append([(token_start, token_stop)])
                    token_accumulator = []
                else:
                    # Add as new word
                    grouped_tokens.append(token_accumulator)
                    token_accumulator = [(token_start, token_stop)]
                last_stop = token_stop
        grouped_tokens.append(token_accumulator)
        grouped_tokens = [x for x in grouped_tokens if x]
        return [(self.text[np.min(token_list):np.max(token_list)], token_list) for token_list in grouped_tokens]

    def __extract_words(self):
        return [w for w, t in self.word_tokens]

    def get_with_masked(self, mask_word_indices: Union[List[WordIdx], WordIdx]):
        """ [MASK] all words by given indices `mask_word_indices` """
        if isinstance(mask_word_indices, (int, np.integer)):
            mask_word_indices = [mask_word_indices]
        result = self.text

        # changing indices must not change other indices -> sort reversed
        for i in sorted(mask_word_indices, reverse=True):
            w, tl = self.word_tokens[i]
            start, stop = np.min(tl), np.max(tl)
            result = replace_with_mask(result, start, stop)
        return result

    def get_with_additional_mask(self, mask_word_indices: Union[List[WordIdx], WordIdx]):
        if isinstance(mask_word_indices, (int, np.integer)):
            mask_word_indices = [mask_word_indices]

        result = list(self.words)

        for i in sorted(mask_word_indices, reverse=True):
            result.insert(i, "[MASK]")
        # return tokenizer.clean_up_tokenization(" ".join(result))
        return " ".join(result)

    def replace_word(self, word_idx, predicted_token):
        result = list(self.words)
        result[word_idx] = predicted_token
        return " ".join(result)

    def replace_mask(self, word_idx, predicted_token):
        assert self.words[word_idx] == "[MASK]"
        return self.replace_word(word_idx, predicted_token)

    def calc_sentiment(self):
        # only for testing purposes. using the batched version (calc_sentiment_batch) substantially faster!
        # print("Using batched version calc_sentiment_batch should be faster!")
        return calc_sentiment_batch([self.text])[0]


def calc_sentiment_batch(sen_list: List[str]):
    if len(sen_list) == 0:
        return []
    result = []
    # otherwise batches could lead to out of memory errors, if the sentence length and batch size get to large
    sub_batches = [sen_list[x:x + SUB_BATCH_SIZE] for x in range(0, len(sen_list), SUB_BATCH_SIZE)]
    logger.debug(f"calc sentiment {len(sen_list)}x{len(model_config.sentiment_tokenizer.tokenize(sen_list[0]))}")
    with torch.no_grad():
        for sb in sub_batches:
            gc.collect()
            torch.cuda.empty_cache()
            t = model_config.sentiment_tokenizer.batch_encode_plus(sb, max_length=512, truncation=True, padding=True, return_attention_mask=True)
            input_ids = torch.tensor(t["input_ids"]).to(device)
            attention_mask = torch.tensor(t["attention_mask"]).to(device)
            result_gpu_t = model_config.sentiment_model(input_ids, attention_mask=attention_mask)
            result_cpu = result_gpu_t[0].cpu()
            result_npy = result_cpu.numpy()
            scores = np.exp(result_npy) / np.exp(result_npy).sum(-1, keepdims=True)
            del input_ids, attention_mask, result_gpu_t, result_cpu, result_npy
            gc.collect()
            torch.cuda.empty_cache()
            result.append(scores)
    return np.concatenate(result)

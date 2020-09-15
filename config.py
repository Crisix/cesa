import logging
from typing import List

import torch
from transformers import BertTokenizerFast, BertForMaskedLM, BertForSequenceClassification, AutoTokenizer
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)
info = logger.info

# TYPING DEFINITIONS:
# These are used to describe what kind of index should be passed to a function.
WordIdx = int
TokenIdx = int
TokenIndices = List[TokenIdx]
WordIndices = List[WordIdx]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def gpu_name():
    try:
        return torch.cuda.get_device_name(device=device)
    except AssertionError:
        return None


SCALING_LOW = 5.  # Words lower than this threshold will never be considered
SCALING_HIGH = 15.  # Words higher than this threshold will have a distance of 0

# Split sentences list for seniment calculation in this sub_batch size
# prevents CUDA out of memory errors
SUB_BATCH_SIZE = 100

# CF EVALUTATION g
COST_PER_ADDITIONAL_WORD = 0.3

# one word must be between any two changes, therefore the indices must differ by 2
MIN_DISTANCE_BETWEEN_CHANGED_WORDS = 2

# only consider the current 250 best examples for merging
# otherwise the problem gets intractable fast.
MAX_EXAMPLES_TO_CONSIDER_FOR_MERGING = 250

# switch between two versions of calculating sentence relevances when texts are split.
# True:  The whole text is used to compute gradients, these gradients are sepearted to
#        their respective sentence.
#        The sentences with highest gradients are considered first.
# False: (Preferred way!) Calculate distances of all sentences polarity to the wanted
#        polarity. Consider sentences with high distance first.
USE_GRADIENTS_FOR_SENTENCE_RELEVANCE = False

MBS_DEPTH = 3
MBS_BEAMS = 2

logger.warning(f"DEVICE={gpu_name()}")


class PlaceholderModel:

    def __getattr__(self, item):
        raise RuntimeError("models and tokenizers are not initialized automatically. "
                           "call load_models(dataset) before searching for counterfactuals")


class Configuration:

    def __init__(self):
        self.model = PlaceholderModel()
        self.tokenizer = PlaceholderModel()
        self.sentiment_model = PlaceholderModel()
        self.sentiment_tokenizer = PlaceholderModel()
        self.perplexity_model = PlaceholderModel()
        self.perplexity_tokenizer = PlaceholderModel()
        self.MODELS_ARE_UNCASED = PlaceholderModel()

    def load(self, counterfactual_model, evalution_model="gpt2-medium"):
        counterfactual_model = counterfactual_model.strip().lower()

        self.MODELS_ARE_UNCASED = True
        if counterfactual_model == "sst-2" or counterfactual_model == "sst2":
            bert_mlm_name = 'bert-large-uncased'
            base_classifier_name = 'textattack/bert-base-uncased-SST-2'
        elif counterfactual_model == "imdb":
            bert_mlm_name = 'bert-large-uncased'
            base_classifier_name = 'textattack/bert-base-uncased-imdb'
        elif counterfactual_model == "ag_news":
            bert_mlm_name = 'bert-large-uncased'
            base_classifier_name = 'textattack/bert-base-uncased-ag-news'
        elif counterfactual_model == "german":
            self.MODELS_ARE_UNCASED = False
            bert_mlm_name = 'bert-base-german-cased'
            base_classifier_name = 'oliverguhr/german-sentiment-bert'
        else:
            assert False, "invalid dataset name"

        self.tokenizer = BertTokenizerFast.from_pretrained(bert_mlm_name)
        self.model = BertForMaskedLM.from_pretrained(bert_mlm_name).to(device)
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(base_classifier_name)
        self.sentiment_model = BertForSequenceClassification.from_pretrained(base_classifier_name).to(device)

        if evalution_model is not None:
            self.perplexity_model = GPT2LMHeadModel.from_pretrained(evalution_model).to(device)
            self.perplexity_tokenizer = GPT2TokenizerFast.from_pretrained(evalution_model)


model_config = Configuration()

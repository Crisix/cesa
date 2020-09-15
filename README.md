This repository contains the corresponding code for my bachelor thesis:

## The generation of <ins>c</ins>ounterfactual <ins>e</ins>xplanations in the context of <ins>s</ins>entiment <ins>a</ins>nalysis

This work explores how the concept of Counterfactual Explanations can be applied to textual data.
For this goal, state-of-the-art machine learning models and text representations are used to find out 
which minimal changes in a text cause the intention of the text to change in a desired direction.
This is made difficult by the high dimensionality of texts and their possibilities to change.
The fragile plausibility of texts also represents a hurdle to be overcome here, since changing a single word can quickly destroy the meaning and grammatical structure of the sentence.
The task chosen here is the sentiment analysis, in which the texts are classified based on the polarity (positivity / negativity of the text).
For a given text an alternative text is to be found, which is still plausible and as similar as possible to the original text, while at the same time the polarity changes in a certain direction.

More details can be found in the [thesis.pdf](thesis.pdf) TODO.

## Usage

### Installation

```shell script
git clone https://github.com/crisix/cesa
cd cesa
pip3 install -r requirements.txt
```

### Example Usage

```python
from config import model_config
from generate_counterfactuals import generate_counterfactuals
from search_utils.Query import Query

negative, positive = [1., 0.], [0., 1.]

# use the BERT model trained on the imdb dataset.
model_config.load("imdb")

text = "A nice movie with lots of good humor."  # Insert your text here.
query = Query(wanted_cls=negative,    # what classification should the generated sentence have?
              mini_beam_search=True,  # replace multiple relevant and adjacent words together?
              consider_top_k=15,      # how many different words of the original text should be considered?
              consider_max_words=500, # how many alternative words should be considered for one word?
              allow_splitting=True,   # can long texts be split into their sentences?
              num_needed=3)           # how many counterfactual explanations are searched?

result = generate_counterfactuals(text, query)

print(result.info())
```

### Results of above example

<pre>
1. a <b>bad</b> movie with lots of good <b>actors</b>.
2. <b>no</b> nice movie with lots of good humor.
3. a nice movie with <b>none</b> of good humor.
</pre>


## Explore results

The generated counterfactual explanations used for the evaluation can be viewed [here](https://crisix.github.io/cesa/).

## Repository Overview

[evaluation](/evaluation) contains the scripts used to generate the counterfactual explanation on the different datasets. + jupyter notebook

[search_utils](/search_utils) contains the data structures for the textual representation and search algorithms for the generation of counterfactual explanations. 

[visualizations](/visualizations) contains the code used to generate visualizations, that are used in the thesis.

[docs](/docs) contains the data for the [exploration website](https://crisix.github.io/cesa/). 

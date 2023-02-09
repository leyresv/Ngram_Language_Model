# Ngram Language Model


## Installation

```bash
pip install git+https://github.com/leyresv/Ngram_Language_Model.git
pip install -r requirements.txt
```

## Usage

To use the Language Model on your own data, open a terminal prompt on the root directory and introduce the following command:
```bash
python src/main/main.py
```


## Model


### Markov assumption

We can compute probabilities of entire sequences using the chain rule of probability:

$$ P(w_1,...,w_n) = P(w_1)P(w_2|w_1)P(w_3|w_1,w_2)...P(w_n|w_1,...,w_{n-1}) = \prod^n_{i=1}P(w_i|w_1,...,w_{i-1})$$

A language model computes the probabilities of sequences of words happening in a certain language. However, applying the chain rule to compute those probabilities would lead to all the sentences non present in the train set to having a 0 probability. To overcome this problem, we can use an n-gram model with the Markov assumption (the probability of a word depends only on the previous word): instead of computing the probaiblity of a word given its entire history (all the previous words in the sentence), we will approximate its conditional probability of the preceding N words:

$$ P(w_n|w_1,...,w_{n-1}) \approx P(w_n|w_{n-N+1},...,w_{n-1})  $$

We can then compute the probability of a full sequence:

$$ P(w_1,...,w_n) \approx \prod^n_{i=1}P(w_i|w_{i-N+1},...,w_{i-1}) $$

To avoid computational issues, we compute instead the log probability:

$$ log(P(w_1,...,w_n)) \approx \sum^n_{i=1}log(P(w_i|w_{i-N+1},...,w_{i-1})) $$


### Maximum Likelihood Estimation

We can estimate the N-gram probabilities by getting counts of all the words appearing in a corpus and normalizing them:

$$ P(w_n|w_1,...,w_{n-1}) = \frac {Count(w_{n-N+1},...,w_{n-1}, w_n)}{Count(w_{n-N+1},...,w_{n-1})}  $$


### Smoothing Methods

There are different options to keep a language model from assigning zero probabilities to unseen contexts:


#### Add-k smoothing:   

We assign a small probability mass to the unseen n-grams:

$$P(w_n|w_1,...,w_{n-1}) = \frac {Count(w_{n-N+1},...,w_{n-1}, w_n) + k}{Count(w_{n-N+1},...,w_{n-1}) + k \cdot V}$$

*   $k$: smoothing factor
*   $V$: vocabulary size


#### Backoff:

We use the n-gram if the evidence is sufficient, otherwise the n-1-gram, otherwise the n-2-gram,..., otherwise the unigram.

#### Interpolation:

We mix the probability estimates from all the n-gram estimators:

$$ \eqalign{
P(w_n|w_1,...,w_{n-N+1}) = &  \lambda_1 \cdot P(w_n) \\
& + \lambda_2 \cdot P(w_n|w_{n-1}) \\
& + \lambda_3 \cdot P(w_n|w_{n-2},w_{n-1}) \\
& + ... \\
& + \lambda_N \cdot P(w_n|w_{n-N+1},...,w_{n-1})  
} $$

With: $\sum_i \lambda_i = 1$

### Evaluation: Perplexity

We use a probability-based metric to evaluate our language model: the perplexity, computed as the inverse probability of the test set, normalized by the number of words:

$$\eqalign{
Perplexity(w_1,...,w_n) = & P(w_1,...,w_n)^{-\frac{1}{n}} \\
& = \sqrt[n]{\frac{1}{P(w_1,...,w_n)}} \\
& = \sqrt[n]{\prod^n_{i=N+1}\frac{1}{P(w_i|w_{i-N},...,w_{i-1})}}
}$$ 





## References

https://web.stanford.edu/~jurafsky/slp3/3.pdf

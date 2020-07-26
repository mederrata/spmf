# (S)parsley-encoded (P)oisson (M)atrix (F)actorization

Implemented using Tensorflow-probability.

This method differs from conventional hierarchical Poisson Matrix factorization methods primarily by sparsifying the encoding transformation rather than the decoding transformation.
The encoding transformation is what computes a representation conditional on data. The decoding transformation takes the representation and produces predictive probability densities.
By sparsifying the encoding, we make each representation coordinate a linear combination of a subset of original data features.
Hence,  inequalities placed on the representation transform directly and transparently into inequalities over the original features.

## Installation

Using pip:
```
pip install git+https://github.com/mederrata/spmf.git
```

## Examples

1. Factorization of random noise [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spmf/blob/master/notebooks/factorizing_random_noise.ipynb)

2. Factorization of synthetic data with underlying linear structure [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mederrata/spmf/blob/master/notebooks/factorize_linear_structure.ipynb)

3. Factorization of synthetic data with underlying nonlinear structure [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mederrata/spmf/blob/master/notebooks/factorize_nonlinear_structure.ipynb)

## Factorizing CSV files

We have included a script that you might find to be useful. It is installed into your `PATH` when using pip.

Usage:

```bash
usage: factorize_csv.py [-h] [-f [CSV_FILE]] [-e [EPOCH]] [-d [DIMENSION]]
                        [-b [BATCH_SIZE]] [-lr [LEARNING_RATE]]
                        [-c [CLIP_VALUE]] [-lt] [-rn]

Train PMF on CSV-formatted count matrix

optional arguments:
  -h, --help            show this help message and exit
  -f [CSV_FILE], --csv-file [CSV_FILE]
                        Enter the CSV file
  -e [EPOCH], --epoch [EPOCH]
                        Enter Epoch value: Default: 300
  -d [DIMENSION], --dimension [DIMENSION]
                        Enter embedding dimension. Default: 2
  -b [BATCH_SIZE], --batch-size [BATCH_SIZE]
                        Enter batch size. Default: 5000
  -lr [LEARNING_RATE], --learning-rate [LEARNING_RATE]
                        Enter float. Default: 0.01
  -c [CLIP_VALUE], --clip-value [CLIP_VALUE]
                        Gradient clip value. Default: 3.0
  -lt, --log-transform  Log-transform?
  -rn, --row-normalize  Row normalize based on counts?
```
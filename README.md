# Male Female Names
I train different models to predict the male-female binary classification with a dataset of 2000 names.

## Datasets
The male names can be found [here](https://www.pampers.co.uk/pregnancy/baby-names/article/top-baby-names-for-boys).</br>
The female names can be found [here](https://www.goodhousekeeping.com/life/parenting/a37668901/top-baby-girl-names/).

## Models
### Linear
A simple fully connected network with 3 hidden layers, with trainable character and positional embeddings.

### Self Attention
An additional self-attention module prepending the linear model.
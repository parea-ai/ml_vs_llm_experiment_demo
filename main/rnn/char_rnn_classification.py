"""
NLP From Scratch: Classifying Names with a Character-Level RNN
**************************************************************
**Author**: `Sean Robertson <https://github.com/spro>`_

We will be building and training a basic character-level Recurrent Neural
Network (RNN) to classify words. This tutorial, along with two other
Natural Language Processing (NLP) "from scratch" tutorials
:doc:`/intermediate/char_rnn_generation_tutorial` and
:doc:`/intermediate/seq2seq_translation_tutorial`, show how to
preprocess data to model NLP. In particular these tutorials do not
use many of the convenience functions of `torchtext`, so you can see how
preprocessing to model NLP works at a low level.

A character-level RNN reads words as a series of characters -
outputting a prediction and "hidden state" at each step, feeding its
previous hidden state into each next step. We take the final prediction
to be the output, i.e. which class the word belongs to.

Specifically, we'll train on a few thousand surnames from 18 languages
of target, and predict which language a name is from based on the
spelling:

.. code-block:: sh

    $ python predict.py Hinton
    (-0.47) Scottish
    (-1.52) English
    (-3.57) Irish

    $ python predict.py Schmidhuber
    (-0.19) German
    (-2.48) Czech
    (-2.68) Dutch


Recommended Preparation
=======================

Before starting this tutorial it is recommended that you have installed PyTorch,
and have a basic understanding of Python programming language and Tensors:

-  https://pytorch.org/ For installation instructions
-  :doc:`/beginner/deep_learning_60min_blitz` to get started with PyTorch in general
   and learn the basics of Tensors
-  :doc:`/beginner/pytorch_with_examples` for a wide and deep overview
-  :doc:`/beginner/former_torchies_tutorial` if you are former Lua Torch user

It would also be useful to know about RNNs and how they work:

-  `The Unreasonable Effectiveness of Recurrent Neural
   Networks <https://karpathy.github.io/2015/05/21/rnn-effectiveness/>`__
   shows a bunch of real life examples
-  `Understanding LSTM
   Networks <https://colah.github.io/posts/2015-08-Understanding-LSTMs/>`__
   is about LSTMs specifically but also informative about RNNs in
   general

Preparing the Data
==================

.. note::
   Download the data from
   `here <https://download.pytorch.org/tutorial/data.zip>`_
   and extract it to the current directory.

Included in the ``data/names`` directory are 18 text files named as
``[Language].txt``. Each file contains a bunch of names, one name per
line, mostly romanized (but we still need to convert from Unicode to
ASCII).

We'll end up with a dictionary of lists of names per language,
``{language: [names ...]}``. The generic variables "category" and "line"
(for language and name in our case) are used for later extensibility.
"""
import string

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch

from main.rnn.rnn import random_training_example, evaluate, RNN, train_iterations, predict
from main.utils import category_from_output, get_category_lines

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)
n_hidden = 128
all_categories, category_lines = get_category_lines()


######################################################################
# Evaluating the Results
# ======================
#
# To see how well the network performs on different categories, we will
# create a confusion matrix, indicating for every actual language (rows)
# which language the network guesses (columns). To calculate the confusion
# matrix a bunch of samples are run through the network with
# ``evaluate()``, which is the same as ``train()`` minus the backprop.
#
def confusion_matrix_evaluate(rnn, all_categories, category_lines, n_categories):
    # Keep track of correct guesses in a confusion matrix
    confusion = torch.zeros(n_categories, n_categories)
    n_confusion = 10000

    # Go through a bunch of examples and record which are correctly guessed
    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = random_training_example(all_categories, category_lines)
        output = evaluate(rnn, line_tensor)
        guess, guess_i = category_from_output(output, all_categories)
        category_i = all_categories.index(category)
        confusion[category_i][guess_i] += 1

    # Normalize by dividing every row by its sum
    for i in range(n_categories):
        confusion[i] /= confusion[i].sum()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()


# predict('Dovesky')
# predict('Jackson')
# predict('Satoshi')

######################################################################
# The final versions of the scripts `in the Practical PyTorch
# repo <https://github.com/spro/practical-pytorch/tree/master/char-rnn-classification>`__
# split the above code into a few files:
#
# -  ``data.py`` (loads files)
# -  ``model.py`` (defines the RNN)
# -  ``train.py`` (runs training)
# -  ``predict.py`` (runs ``predict()`` with command line arguments)
# -  ``server.py`` (serve prediction as a JSON API with ``bottle.py``)
#
# Run ``train.py`` to train and save the network.
#
# Run ``predict.py`` with a name to view predictions:
#
# .. code-block:: sh
#
#     $ python predict.py Hazaki
#     (-0.42) Japanese
#     (-1.39) Polish
#     (-3.51) Czech
#
# Run ``server.py`` and visit http://localhost:5533/Yourname to get JSON
# output of predictions.
#


######################################################################
# Exercises
# =========
#
# -  Try with a different dataset of line -> category, for example:
#
#    -  Any word -> language
#    -  First name -> gender
#    -  Character name -> writer
#    -  Page title -> blog or subreddit
#
# -  Get better results with a bigger and/or better shaped network
#
#    -  Add more linear layers
#    -  Try the ``nn.LSTM`` and ``nn.GRU`` layers
#    -  Combine multiple of these RNNs as a higher level network
#
def set_up(n_iters=100000, show_confusion_matrix=True):
    n_categories = len(all_categories)
    rnn = RNN(n_letters, n_hidden, n_categories)
    if show_confusion_matrix:
        confusion_matrix_evaluate(rnn, all_categories, category_lines, n_categories)
    train_iterations(rnn, all_categories, category_lines, n_iters)
    return rnn


if __name__ == '__main__':
    rnn = set_up()
    print(predict(rnn, all_categories, input_line='Dovesky', n_predictions=1))
    print(predict(rnn, all_categories, input_line='Satoshi', n_predictions=1))
    print(predict(rnn, all_categories, input_line='Jackson', n_predictions=1))

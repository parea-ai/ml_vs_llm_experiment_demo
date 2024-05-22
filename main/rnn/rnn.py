import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from main.utils import random_choice, line_to_tensor, category_from_output, time_since


class RNN(nn.Module):
    ######################################################################
    # Creating the Network
    # ====================
    #
    # Before autograd, creating a recurrent neural network in Torch involved
    # cloning the parameters of a layer over several timesteps. The layers
    # held hidden state and gradients which are now entirely handled by the
    # graph itself. This means you can implement a RNN in a very "pure" way,
    # as regular feed-forward layers.
    #
    # This RNN module implements a "vanilla RNN" an is just 3 linear layers
    # which operate on an input and hidden state, with a ``LogSoftmax`` layer
    # after the output.
    #
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, _input, hidden):
        hidden = F.tanh(self.i2h(_input) + self.h2h(hidden))
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


def random_training_example(all_categories, category_lines):
    category = random_choice(all_categories)
    line = random_choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor


def train(rnn, category_tensor, line_tensor):
    ######################################################################
    # Training the Network
    # --------------------
    #
    # Now all it takes to train this network is show it a bunch of examples,
    # have it make guesses, and tell it if it's wrong.
    #
    # For the loss function ``nn.NLLLoss`` is appropriate, since the last
    # layer of the RNN is ``nn.LogSoftmax``.
    #
    criterion = nn.NLLLoss()
    learning_rate = 0.005  # If you set this too high, it might explode. If too low, it might not learn
    hidden = rnn.init_hidden()
    output = None

    rnn.zero_grad()

    ######################################################################
    # Each loop of training will:
    #
    # -  Create input and target tensors
    # -  Create a zeroed initial hidden state
    # -  Read each letter in and
    #
    #    -  Keep hidden state for next letter
    #
    # -  Compare final output to target
    # -  Back-propagate
    # -  Return the output and loss
    #
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()


def train_iterations(rnn, all_categories, category_lines, n_iters=100000):
    print_every = n_iters / 20
    plot_every = n_iters / 100
    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []

    start = time.time()

    for _iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = random_training_example(all_categories, category_lines)
        output, loss = train(rnn, category_tensor, line_tensor)
        current_loss += loss

        # Print ``iter`` number, loss, name and guess
        if _iter % print_every == 0:
            guess, guess_i = category_from_output(output, all_categories)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print(
                '%d %d%% (%s) %.4f %s / %s %s' % (
                    _iter, _iter / n_iters * 100, time_since(start), loss, line, guess, correct)
            )

        # Add current loss avg to list of losses
        if _iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0


def evaluate(rnn, line_tensor):
    hidden = rnn.init_hidden()
    output = None
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output


def predict(rnn, all_categories, input_line, n_predictions=3, verbose=True):
    if verbose:
        print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(rnn, line_to_tensor(input_line))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            if verbose:
                print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])
    return predictions

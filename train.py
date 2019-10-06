from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import random

import re
import os
import sys
import json
import time
import math

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

import process_data
import seq2seq

SOS_token = 0
EOS_token = 1
# Teacher forcing feeds the decoder the target sequence as opposed to the decoder's previous
# (potentially) incorrect decision. This can lead to improvements in accuracy but excessive
# force can lead to a more instable, incoherent network
teacher_forcing_ration = 0.5

def train(input_tensor, target_tensor, encoder, decoder,
        encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):

    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    output_length = input_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device)

    loss = 0

    for e_i in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[e_i], encoder_hidden)
        encoder_output[e_i] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    for d_i in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)

        if use_teacher_forcing:
            # Feed the target sequence as next input
            loss += criterion(decoder_output, tensor_target[d_i])
            decoder_input = tensor_target[d_i]
        else:
            top_v, top_i = decoder_output.topk(1)
            decoder_input = top_i.squeeze().detach()
            loss += criterion(decoder_output, target_tensor[d_i])

            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


# Generic helper methods for how much time has elapsed
def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))



def train_iters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_total_losses = 0  # Will get reset every print_every
    plot_total_losses = 0   # Will get reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    training_pairs = [tensorsFromPair(random.choice(pairs)) for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]


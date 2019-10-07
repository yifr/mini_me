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
MAX_LENGTH = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Teacher forcing feeds the decoder the target sequence as opposed to the decoder's previous
# (potentially) incorrect decision. This can lead to improvements in accuracy but excessive
# force can lead to a more instable, incoherent network
teacher_forcing_ratio = 0.5

def train(input_tensor, target_tensor, encoder, decoder,
        encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):

    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for e_i in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[e_i], encoder_hidden)
        encoder_outputs[e_i] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    for d_i in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)

        if use_teacher_forcing:
            # Feed the target sequence as next input
            loss += criterion(decoder_output, target_tensor[d_i])
            decoder_input = target_tensor[d_i]
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



def train_iters(encoder, decoder, n_iters, messages, responses, data,
        print_every=100, plot_every=1000, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Will get reset every print_every
    plot_loss_total = 0   # Will get reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    training_pairs = [process_data.tensors_from_pair(random.choice(data), messages, responses) for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder, decoder,
                     encoder_optimizer, decoder_optimizer, criterion)

        # print('Training Iteration: ' + str(iter) + ' => loss = ' + str(loss))
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (time_since(start, iter / n_iters),
                iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            show_plot(plot_losses, iter)

def show_plot(points, i):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.savefig('training/iter_' + str(i) + '.png')

def evaluate(encoder, decoder, input_lang, output_lang, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = process_data.tensor_from_sentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(max_length, encoder_hidden.size(), device=device)

        for e_i in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[e_i], encoder_hidden)
            encoder_output[e_i] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for d_i in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[d_i] = decoder_attention.data
            top_v, top_i = decoder_output.data.topk(1)
            if top_i.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[top_i.item()])

            decoder_input = top_i.squeeze().detach()

    return decoded_words, decoder_attentions[:di + 1]

    def evaluateRandomly(pairs, encoder, decoder, n=10):
        for i in range(n):
            pair = random.choice(pairs)
            print('>', pair[0])
            print('=', pair[1])
            output_words, attentions = evaluate(encoder, decoder, pair[0])
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
            print('')

def main():
    messages = seq2seq.Lang('messages')
    responses = seq2seq.Lang('responses')
    data = process_data.get_fb_data()
    filtered_data = process_data.filter_pairs(data)

    print('Building vocabulary...')
    for pair in filtered_data:
        msg = pair[0]
        resp = pair[1]
        messages.add_sentence(msg)
        responses.add_sentence(resp)

    print('Unique words counted: ')
    print('{}: {}'.format(messages.name, messages.n_words))
    print('{}: {}'.format(responses.name, responses.n_words))

    hidden_size = 256
    encoder1 = seq2seq.EncoderRNN(messages.n_words, hidden_size).to(device)
    attn_decoder1 = seq2seq.AttnDecoderRNN(hidden_size, responses.n_words, dropout_p=0.1).to(device)

    train_iters(encoder1, attn_decoder1, 75000, messages, responses, filtered_data)

    torch.save(encoder1.state_dict(), 'encoder.pt')
    torch.save(attn_decoder1.state_dict(), 'decoder.pt')

if __name__=='__main__':
    main()

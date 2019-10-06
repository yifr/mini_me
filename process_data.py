from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import random

import re
import os
import sys
import json

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def get_fb_data(data_dir='./inbox', me='Yoni Friedman'):
    '''
    Reads through all messages in facebook data,
    returns dictionary with message response pairs,
    (message = sent by someone else, response = sent by you)
    and writes message, response pairs to output txt file
    '''
    # Grab all file paths of message conversations
    msg_files = []
    for root, dirs, files in os.walk(data_dir):
        path = root.split(os.sep)
        for fname in files:
            if 'message' in fname:
                msg_files.append(os.path.join(root, fname))


    # Parse json information
    dataset = open('dataset.txt', 'a+', encoding='utf-8')
    data_dict = dict()

    print('Parsing message data...')
    for msg_file in msg_files:
        # Load json data
        f = open(msg_file)
        data = json.load(f)
        f.close()

        messages = data['messages']
        messages.reverse()          # Read messages from oldest to newest

        prev_sender = ""
        curr_sender = ""
        message = ""
        response = ""
        for msg in messages:
            if not msg.get('content'):  # Someone sent a sticker or a file - exclude these
                continue

            curr_sender = msg['sender_name']
            # Skip first message if I sent it (our bot is purely responsive)
            if not prev_sender and curr_sender == me:
                continue

            elif curr_sender == me:
                response += msg['content'] + '\n'
            else:
                message += msg['content'] + '\n'

            # Write message response pair when I've completed response (next message being sent)
            if curr_sender != me and prev_sender == me and message and response:
                data_dict[message] = response
                dataset.write(message + '\t' + response)

                message = ''
                response = ''

            prev_sender = curr_sender

    print('Parsed {} message threads...'.format(len(msg_files)))
    dataset.close()
    return data_dict


class EncoderRNN():
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.LSTM = nn.LSTM(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.LSTM(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnDecoderRNN():
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.LSTN = nn.LSTM(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden_size, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.LSTM(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def main():
    data = get_fb_data()
    messages = Lang('messages')
    responses = Lang('responses')
    print('Building vocabulary...')
    for msg, resp in data.items():
        messages.add_sentence(msg)
        responses.add_sentence(resp)

    print('Unique words counted: ')
    print('{}: {}'.format(messages.name, messages.n_words))
    print('{}: {}'.format(responses.name, responses.n_words))

if __name__=='__main__':
    main()

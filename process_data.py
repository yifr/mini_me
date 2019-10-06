from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import random

import re
import os
import sys
import json
import process_data

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


SOS_token = 0
EOS_token = 1

'''
Initial data processing:
'''
def get_fb_data(data_dir='./inbox', target='Yoni Friedman', write_data=False):
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
    if write_data:
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
            if not prev_sender and curr_sender == target:
                continue

            elif curr_sender == target:
                response += msg['content'] + '\n'
            else:
                message += msg['content'] + '\n'

            # Write message response pair when I've completed response (next message being sent)
            if curr_sender != target and prev_sender == target and message and response:
                data_dict[message] = response
                if write_data:
                    dataset.write(message + '\t' + response)

                message = ''
                response = ''

            prev_sender = curr_sender

    print('Parsed {} message threads...'.format(len(msg_files)))
    if write_data:
        dataset.close()
    return data_dict


'''
Preparing training data:
'''
def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensor_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensors_from_pair(pair):
    input_tensor = tensor_from_sentence('messages', pair[0])
    target_tensor = tensor_from_sentence('responses', pair[1])
    return (input_tensor, target_tensor)


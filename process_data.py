import re
import os
import sys
import json

def get_fb_data(msg_file="message", me="Yoni Friedman"):
    # Get paths for all message json files
    msg_files = []
    for root, dirs, files in os.walk('./inbox'):
        path = root.split(os.sep)
        for fname in files:
            if msg_file in fname:
                msg_files.append(os.path.join(root, fname))


    # Parse json information
    dataset = open('dataset.txt', 'a+')
    for mf in msg_files:
        f = open(mf)
        data = json.load(f)
        f.close()

        messages = data['messages']

        previous_sender = messages[0]['sender_name']
        message = ""
        response = ""

        for msg in messages:
            current_sender = msg.get('sender_name')
            if current_sender == me:
                # Previous sender just finished talking and we're sending a message - write their message to dataset
                if previous_sender != me:
                    dataset.write('message: ' + message + '\n')
                    message = ''

                if msg.get('content'):
                    response += msg.get('content') + '\n'

            # Target finished responding and friend is sending message - write target response to dataset
            elif previous_sender == me:
                dataset.write('response: ' + response + '\n')
                response = ""
                if msg.get('content'):
                    message += msg.get('content') + '\n'

            # Friend(s) are still sending messages
            elif msg.get('content'):
                message += msg.get('content') + '\n'

            # update previous sender
            previous_sender = current_sender

        # Write any remaining content to dataset
        if message:
            dataset.write('message: ' + message + '\n')
        if response:
            dataset.write('response: ' + response + '\n')

    dataset.close()

def main():
    get_fb_data('message_')

if __name__=='__main__':
    main()

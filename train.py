import json
from spacy_utils import tokenize, lemmatize, bag_of_words 
import numpy as np

import torch

with open ('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))
        
ignore_words = ['?', '.', '!', ',']
all_words = [lemmatize(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tag = sorted(tags)

X_train = []
y_train = []
for (pattern_sequence, tag) in xy:
    bag = bag_of_words(pattern_sequence, all_words)
    X_train.append(bag)
    
    label = tags.index(tag)
    y_train.append(label)
    
X_train = np.array(X_train)
y_train = np.array(y_train)
    
    

    
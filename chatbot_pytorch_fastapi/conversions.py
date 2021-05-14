# -*- coding: utf-8 -*-
"""
Created on Fri May 14 17:17:39 2021

@author: Guru Aathavan AL U
"""

import pickle
import numpy as np
import torch

status=torch.cuda.is_available()
#status=False

def pad_features(reviews_ints, seq_length=20):
    features = np.zeros((len(reviews_ints), seq_length), dtype=int)
    for i, row in enumerate(reviews_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]
    return features

#all dictionaries for word to int and vice versa conversion
f=open("word_to_int.pkl","rb")
word_to_int=pickle.load(f)

f=open("int_to_word.pkl","rb")
int_to_word=pickle.load(f)

f=open("label_to_int.pkl","rb")
label_to_int=pickle.load(f)

f=open("int_to_label.pkl","rb")
in_to_label=pickle.load(f)

f=open("label_to_response.pkl","rb")
labe_to_response=pickle.load(f)



def convert(question):
    question=question.split()
    q=[]
    for i in question:
        q.append(word_to_int[i])
    q=np.array(q)
    print(q)
    inputs=torch.Tensor(pad_features([q]))
    if status:
        inputs=inputs.cuda()
    inputs=torch.tensor(inputs).to(torch.int64)
    return inputs



def device():
    return status

def int_to_label(num):
    return (in_to_label[num])

def label_to_response(label):
    return (labe_to_response[label])
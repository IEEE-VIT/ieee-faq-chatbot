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

def preprocess(text):

    # Replace punctuation with tokens so we can use them in our model
    text = text.lower()
    text = text.replace('.', '')
    text = text.replace(',','')
    text = text.replace('"', '')
    text = text.replace(';', '')
    text = text.replace('!', '')
    text = text.replace('?', '')
    text = text.replace('(', '')
    text = text.replace(')', '')
    text = text.replace('--', '')
    text = text.replace('?', '')
    text = text.replace('\n', '')
    text = text.replace(':', '')
    
    return text

def convert(question):
    question=preprocess(question).split()
    q=[]
    with open("word_to_int.pkl","rb") as f:
        word_to_int=pickle.load(f)
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
    with open("int_to_label.pkl","rb") as f:
        in_to_label=pickle.load(f)
    return (in_to_label[num])


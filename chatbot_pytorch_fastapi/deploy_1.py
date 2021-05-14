# -*- coding: utf-8 -*-
"""
Created on Sat May  8 12:20:28 2021

@author: Guru Aathavan AL U
"""  

import pickle
import torch 
from fastapi import FastAPI
import uvicorn
from torch import nn
#import re
import numpy as np

app=FastAPI()
device="cuda"
train_on_gpu=True


#all dictionaries for word to int and vice versa conversion
f=open("word_to_int.pkl","rb")
word_to_int=pickle.load(f)
1
f=open("int_to_word.pkl","rb")
int_to_word=pickle.load(f)

f=open("label_to_int.pkl","rb")
label_to_int=pickle.load(f)

f=open("int_to_label.pkl","rb")
int_to_label=pickle.load(f)

f=open("label_to_response.pkl","rb")
label_to_response=pickle.load(f)

#function to pad the features
def pad_features(reviews_ints, seq_length=20):
    # getting the correct rows x cols shape
    features = np.zeros((len(reviews_ints), seq_length), dtype=int)
    # for each review, I grab that review and 
    for i, row in enumerate(reviews_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]
    return features


#features of the network
vocab_size = 4163
output_size = 27
embedding_dim = 300
hidden_dim = 256
n_layers = 1

#Semantic classifier that is used for classifying intent of the question.
class Semantic_Classifier(nn.Module):
   

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        super(Semantic_Classifier, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                            dropout=drop_prob, batch_first=True)
        
        # dropout layer
        self.dropout = nn.Dropout(0.3)
        
        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim*2, output_size)
        self.sig = nn.Sigmoid()
        

    def forward(self, x, hidden):
       
        batch_size = x.size(0)
        embeds = self.embedding(x)
        
        lstm_out, hidden = self.lstm(embeds, hidden)
        out = self.fc(torch.cat((torch.squeeze(hidden[0]),torch.squeeze(hidden[1])),0))
       
        sig_out = self.sig(out)
        sig_out = sig_out.view(batch_size, -1)
        return sig_out, hidden
    
    
    def init_hidden(self, batch_size):
        
        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden

#net1 with fetures defined above
net_1=Semantic_Classifier(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
net_1.cuda()
net_1.load_state_dict(torch.load("ieeesite_v2.pt"))
net_1.eval()
val_h=net_1.init_hidden(1)

def convert(question):
    question=question.split()
    q=[]
    for i in question:
        q.append(word_to_int[i])
    q=np.array(q)
    print(q)
    inputs=torch.Tensor(pad_features([q])).cuda()
    inputs=torch.tensor(inputs).to(torch.int64)
    return inputs

@app.get("/")
def index():
    return {"message": "stranger"}


@app.get("/Welcome")
def get_name(name:str):
    return{'trail':f'{name}'}

@app.post('/predict')
def predict_intent(data:str):
    data=convert(data)
    hidden=val_h
    _,output=torch.topk(net_1(data,hidden)[0],1)
    output=label_to_response[int_to_label[output.item()]]
    
    return{
        'prediction':output
        }

if __name__=="main":
    uvicorn.run(app,host='127.0.0.1',port=8000)
    
    
"""
@app.get("/predict/{sentence}")
def predict(sentence: str):
    pred = prediction(sentence)[0][0]
    return {"message": str(pred)}
"""
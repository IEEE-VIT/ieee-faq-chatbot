# -*- coding: utf-8 -*-
"""
Created on Fri May 14 17:11:30 2021

@author: Guru Aathavan AL U
"""

import torch
from torch import nn
import numpy as np
import time
import conversions

status=torch.cuda.is_available()
#status=False

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
        
        if (status):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden

#net1 with fetures defined above
net_1=Semantic_Classifier(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
if status:
    net_1.cuda()
net_1.load_state_dict(torch.load("ieeesite_v2.pt"))
net_1.eval()
val_h=net_1.init_hidden(1)

def custom_forward(data):
    data=conversions.convert(data)  
    
    hidden=val_h
    _,output=torch.topk(net_1(data,hidden)[0],1)
    
    label=conversions.int_to_label(output.item())
    output=conversions.label_to_response(label)
    #print(val_h)
    return label,output
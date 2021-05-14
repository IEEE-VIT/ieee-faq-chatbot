# -*- coding: utf-8 -*-
"""
Created on Fri May 14 17:37:18 2021

@author: Guru Aathavan AL U
"""

from fastapi import FastAPI
import uvicorn
import time
import model
import conversions

app=FastAPI()


@app.get("/")
def index():
    return {"IEEE Chatbot": "v2"}

@app.get("/trail")
def get_name(name:str):
    return{'trail':f'{name}'}

@app.post('/predict')
def predict_intent(data:str):
    res_time=time.time()
    query=data
    #data=conversions.convert(data)    
    """
    hidden=val_h
    _,output=torch.topk(net_1(data,hidden)[0],1)
    label=int_to_label[output.item()]
    output=label_to_response[int_to_label[output.item()]]
    """
    label,output=model.custom_forward(data)
    res_time=time.time()-res_time
    device=conversions.device()
    
    return{
        'query':query,
        'label':label,
        'response':output,
        'cuda':device,
        'response time ':res_time
        }

if __name__=="main":
    uvicorn.run(app,host='127.0.0.1',port=8000)
    

import os
import uvicorn
import time
from dotenv import load_dotenv
from pymongo import MongoClient
from pydantic import BaseModel 
from typing import Optional
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from tensorflow_src.tensorflow_model import ieee_faq_bot
from pytorch_src.pytorch_model import custom_forward
from user_output import user_output
from fastapi.middleware.cors import CORSMiddleware
from profanity_check import predict

load_dotenv()
mongo_uri = os.getenv("mongo_uri")
client = MongoClient(mongo_uri)
db=client.get_database("ieee_faq_bot")
queries=db.user_input_queries

app = FastAPI(
    title="IEEE FAQ chatbot",
    description="Answer all frequently asked questions")  

origins=["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def tensorflow_model(user_query):
    # classify the input query as one of the intents using tensorflow
    make_prediction = ieee_faq_bot(user_query)
    return make_prediction 

def pytorch_model(user_query):
    # classify the input query as one of the intents using pytorch
    make_prediction = custom_forward(user_query)
    return make_prediction

def prof_check(user_query):
    user_query_=[]
    user_query_.append(user_query)
    check=predict(user_query_)
    if check[0]==1:
       return True
    else:
        return False

#add to mongo
def add_to_mongo(user_query, predicted_intent):
    query_append_status="appended"
    try:
        queries.insert_one(
            {"input_query":user_query,
            "classified_intent":predicted_intent})
        return query_append_status
    except ConnectionError:
        query_append_status = "Something went wrong, not appended"
        return query_append_status

#create request body
class data(BaseModel):
    user_input: str
    model_selection: int

# create app object with title and description 
@app.get('/')
def index():
    message = "IEEE FAQ chatbot is live."
    return message

@app.post('/chatbot',description="enter 1 under model_selection for tensorflow model and 2 for pytorch model")
def chatbot(data:data):
    res_time=time.time()
    data=data.dict()
    # load input query and model by user
    user_query = data['user_input']  
    selected_model = data['model_selection']
    con=prof_check(user_query)

    if con == True:
        predicted_intent="gen_bad"
        model_in_use="Not required"
        query_append_status="Profanity found, not appended"
    else:
        if selected_model == 1:
            predicted_intent=tensorflow_model(user_query)
            model_in_use = "TensorFlow"
        elif selected_model == 2:
            predicted_intent=pytorch_model(user_query)
            model_in_use = "PyTorch"
        elif selected_model == 0:
            predicted_intent=tensorflow_model(user_query)
            model_in_use = "TensorFlow"
        else :
            raise ValueError

    #generate final output for given query
    final_user_output = user_output(predicted_intent)

    #write data to mongoDB
    if predicted_intent == "gen_bad" or predicted_intent == "invalid_query":
        query_append_status = "Not appended, profanity found and/or invalid query"
    else:
        query_append_status = add_to_mongo(user_query, predicted_intent)

    return {  
        'user query': user_query,
        'predicted intent': predicted_intent,
        'output to user': final_user_output,
        'model used' : model_in_use,
        'response time' : time.time()-res_time,
        'mongodb append status' : query_append_status
    }

@app.exception_handler(Exception)
async def validation_exception_handler(request,exc):
    if ValueError:
        return PlainTextResponse("Enter a valid model. 1 for tensorflow and 2 for pytorch under "'model_selection'" in request body", status_code=400)
    else:
        return PlainTextResponse("Something went wrong",status_code=400)

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000, Reload=True)

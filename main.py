import uvicorn
import time
from fastapi import FastAPI
from pydantic import BaseModel 
from fastapi.responses import PlainTextResponse
from tensorflow_src.tensorflow_model import ieee_faq_bot
from user_output import user_output
from pytorch_src.pytorch_model import custom_forward

def tensorflow_model(user_query):
    # classify the input query as one of the intents using tensorflow
    make_prediction = ieee_faq_bot(user_query)
    return make_prediction 

def pytorch_model(user_query):
    # classify the input query as one of the intents using pytorch
    make_prediction = custom_forward(user_query)
    return make_prediction

app = FastAPI(   # create app object with title and description 
    title="IEEE FAQ chatbot",
    description="Answer all frequently asked questions")  

class data(BaseModel):
    user_input: str
    model_selection: int

@app.get('/')
def index():
    message = "IEEE FAQ chatbot is live. Head over to http://127.0.0.1:8000/docs to test it out."
    return message

@app.post('/chatbot',description="enter 1 under model_selection for tensorflow model and 2 for pytorch model")
def chatbot(data:data):
    res_time=time.time()
    data=data.dict()
    # load input query and model by user
    user_query = data['user_input']  
    selected_model = data['model_selection']

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

    return {  
        'user query': user_query,
        'predicted intent': predicted_intent,
        'output to user': final_user_output,
        'model used' : model_in_use,
        'response time' : time.time()-res_time
    }

@app.exception_handler(Exception)
async def validation_exception_handler(request,exc):
    if ValueError:
        return PlainTextResponse("Enter a valid model. 1 for tensorflow and 2 for pytorch under "'model_selection'" in request body", status_code=400)
    else:
        return PlainTextResponse("Something went wrong",status_code=400)

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000, Reload=True)

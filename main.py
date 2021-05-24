import uvicorn
import time
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from user_input import user_input
from tensorflow_model import ieee_faq_bot
from user_output import user_output

def tensorflow_model(user_query):
    # classify the input query as one of the intents using tensorflow
    make_prediction = ieee_faq_bot(user_query)
    return make_prediction 

def pytorch_model(user_query):
    # classify the input query as one of the intents using pytorch
    make_prediction = "gen_bye"
    return make_prediction

# create app object with title and description 
app = FastAPI(
    title="IEEE FAQ chatbot",
    description="Answer all frequently asked questions")  

@app.get('/')
def index():
    message = "IEEE FAQ chatbot is live. Head over to http://127.0.0.1:8000/docs to test it out."
    return message

@app.post('/chatbot',description="enter 1 under model_selection for tensorflow model and 2 for pytorch model")
def chatbot(data: user_input):
    res_time=time.time()
    data = data.dict()

    # load input query and model by user
    user_query = data['query']  
    selected_model = data['model_selection']

    if selected_model == 1:
        predicted_intent=tensorflow_model(user_query)
        model_in_use = "TensorFlow"
    elif selected_model == 2:
        predicted_intent=pytorch_model(user_query)
        model_in_use = "PyTorch"
    else :
        raise Exception("Please enter a valid model")

    #generate final output for given query
    final_user_output = user_output(predicted_intent)

    return {  
        'user query': user_query,
        'predicted intent': predicted_intent,
        'output to user': final_user_output,
        'model used' : model_in_use,
        'response time' : res_time
    }

@app.exception_handler(Exception)
async def validation_exception_handler(request,exc):
    return PlainTextResponse("Enter a valid model. 1 for tensorflow and 2 for pytorch", status_code=400)

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000, Reload=True)

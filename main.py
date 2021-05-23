import uvicorn
from fastapi import FastAPI
from user_input import user_input
from model import ieee_faq_bot
from user_output import user_output

# create app object with title and description 
app = FastAPI(
    title="IEEE FAQ chatbot",
    description="Answer all frequently asked questions")  

@app.get('/')
def index():
    return {"message": "IEEE FAQ chatbot is live"}

@app.post('/chatbot')
def chatbot(data: user_input):
    data = data.dict()
    # load input query by user
    user_query = data['query']  
    # classify the input query as one of the intents
    make_prediction = ieee_faq_bot(user_query)
    # get output based on classified intent
    final_user_output = user_output(make_prediction)

    return {  
        'user query': user_query,
        'prediction': make_prediction,
        'output to user': final_user_output
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

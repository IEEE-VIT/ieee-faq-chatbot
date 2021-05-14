import uvicorn
from fastapi import FastAPI
from user_input import user_input
from FAQ_BOT_MODEL import user
from user_output import user_output
app = FastAPI() #create app object

@app.get('/')
def index():
    return{'message': 'hello from fastapi'}

@app.post('/chatbot')
def chatbot(data:user_input):
    data=data.dict()
    query=data['query']
    prediction=user(query)
    final_user_output=user_output(prediction)
    print(prediction)
    return {
        'user query' : query,
        'prediction' : prediction,
        'output to user': final_user_output
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1',port=8000)
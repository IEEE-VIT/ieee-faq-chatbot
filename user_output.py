import json
import random

def user_output(predicted_intent):   
    extracted_responses=[]
    search=predicted_intent
    with open("output.json") as responses:
        loaded_responses=json.load(responses)
        for i in loaded_responses[search]:      #extract responses for the classified intent
	        extracted_responses.append(i["op"])
    final_user_output=random.choice(extracted_responses)   #randomly give one response as output

    return final_user_output
    
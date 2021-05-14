import json
import random

output = open('output.json',)

# parse x:
y = json.load(output)



def user_output(predicted_intent):
    A=[]
    search=predicted_intent
    for i in y[search]:
	    A.append(i["op"])
    final_user_output=random.choice(A)

    return final_user_output
    
output.close()
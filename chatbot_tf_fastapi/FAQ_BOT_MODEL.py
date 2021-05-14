import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import OneHotEncoder

def load_dataset(filename):
    df = pd.read_csv(filename, encoding = "latin1", names = ["question", "intent"])
    df=df.dropna()
    #print(df.head())
    #print(df.isnull().any())
    intent = df["intent"]
    #unique_intent = list(set(intent))
    question = list(df["question"])
    return (intent, question)

intent, question = load_dataset("raw_data_augmented_final.csv")

unique_intent=['ieee_github', 'ieee_how_long', 'gen_who_made', 'ieee_domains', 'ieee_be_part_who', 'gen_human', 'ieee_past_events', 
'ieee_sponsor', 'gen_help', 'ieee_projects', 'gen_who_you', 'ieee_what_does', 'gen_hi', 'ieee_collab', 'gen_thanks', 
'ieee_other_college', 'ieee_current_board', 'ieee_other_linked', 'ieee_social_media', 'gen_query', 'ieee_what', 'gen_bot', 
'ieee_future_events', 'ieee_further_contact', 'gen_how_you', 'ieee_techloop', 'gen_bye']

def cleaning(question):
    words = []
    for s in question:
        clean = re.sub(r'[^a-z A-Z]', " ", s)
        w = word_tokenize(clean)
        #stemming
        words.append([i.lower() for i in w])
    
    return words  

cleaned_words = cleaning(question)

def create_tokenizer(words, filters = '!"#$%&*+,-./:;<=>?@[\]^`{|}~'):
    token = Tokenizer(filters = filters)
    token.fit_on_texts(words)
    return token

def max_length(words):
    return(len(max(words, key = len)))

word_tokenizer = create_tokenizer(cleaned_words)
#vocab_size = len(word_tokenizer.word_index) + 1
max_length = max_length(cleaned_words)

def encoding_doc(token, words):
    return(token.texts_to_sequences(words))

encoded_doc = encoding_doc(word_tokenizer, cleaned_words)

def padding_doc(encoded_doc, max_length):
    return(pad_sequences(encoded_doc, maxlen = max_length, padding = "post"))

padded_doc = padding_doc(encoded_doc, max_length)

model = load_model("ieee_bot_test1.h5")


def predictions(text):
    clean = re.sub(r'[^ a-z A-Z 0-9]'," ", text)
    test_word = word_tokenize(clean)
    test_word = [w.lower() for w in test_word]
    test_ls = word_tokenizer.texts_to_sequences(test_word)
    #print(test_word)            ##
    #Check for unknown words
    if [] in test_ls:
        test_ls = list(filter(None, test_ls))
    
    test_ls = np.array(test_ls).reshape(1, len(test_ls))
 
    x = padding_doc(test_ls, max_length)
  
    pred = model.predict(x)
  
    return pred 

def get_final_output(pred, classes):
    predictions = pred[0]
 
    classes = np.array(classes)
    ids = np.argsort(-predictions)
    classes = classes[ids]
    predictions = -np.sort(-predictions)
    pred_intent=classes[0]
 
    for i in range(5):
       print("%s has confidence = %s" % (classes[i], (predictions[i])))
    return pred_intent

def user(text):
    #text=input('Enter your query : ')
    pred = predictions(text)
    final_intent=get_final_output(pred,unique_intent)

    return final_intent
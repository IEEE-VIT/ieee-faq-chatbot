import re
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from nltk.tokenize import word_tokenize

def load_dataset(filename):
    #read data from csv file with 2 columns: one for question and other for intent
    df = pd.read_csv(filename, encoding="latin1", names=["question", "intent"])
    df = df.dropna()
    intent = df["intent"]
    questions = list(df["question"])
    return (intent, questions)

intent, questions = load_dataset("augmented_data.csv")

#specify all the unique intents in the data
unique_intent = [
    "ieee_github",
    "ieee_how_long",
    "gen_who_made",
    "ieee_domains",
    "ieee_be_part_who",
    "gen_human",
    "ieee_past_events",
    "ieee_sponsor",
    "gen_help",
    "ieee_projects",
    "gen_who_you",
    "ieee_what_does",
    "gen_hi",
    "ieee_collab",
    "gen_thanks",
    "ieee_other_college",
    "ieee_current_board",
    "ieee_other_linked",
    "ieee_social_media",
    "gen_query",
    "ieee_what",
    "gen_bot",
    "ieee_future_events",
    "ieee_further_contact",
    "gen_how_you",
    "ieee_techloop",
    "gen_bye",
]

def cleaning(question):
    words = []
    #clean the questions of all punctuations
    for word in question:
        clean = re.sub(r"[^a-z A-Z 0-9]", " ", word)
        clean = word_tokenize(clean)
        words.append([i.lower() for i in clean])

    return words

cleaned_words = cleaning(questions)

def create_tokenizer(cleaned_words, filters='!"#$%&*+,-./:;<=>?@[\]^`{|}~'):
    #tokenize the cleaned words in questions upto word level 
    token = Tokenizer(filters=filters)
    token.fit_on_texts(cleaned_words)
    return token

def max_length(cleaned_words):
    #get the number of words in longest question
    return len(max(cleaned_words, key=len))

word_tokenizer = create_tokenizer(cleaned_words)
max_length = max_length(cleaned_words)

def encoding_doc(token, cleaned_words):
    return token.texts_to_sequences(cleaned_words)

encoded_doc = encoding_doc(word_tokenizer, cleaned_words)

def padding_doc(encoded_doc, max_length):
    return pad_sequences(encoded_doc, maxlen=max_length, padding="post")

padded_doc = padding_doc(encoded_doc, max_length)

#load the pre-trained model
loaded_model = load_model("ieee_faq_bot.h5")

def predictions(user_query):
    #clean and tokenize input user query
    clean_user_query = re.sub(r"[^ a-z A-Z 0-9]", " ", user_query)
    clean_user_query = word_tokenize(clean_user_query)
    clean_user_query = [w.lower() for w in clean_user_query]
    tokenized_user_query = word_tokenizer.texts_to_sequences(clean_user_query)

    #filter unknown words
    if [] in tokenized_user_query:
        tokenized_user_query = list(filter(None, tokenized_user_query))

    #reshape array as per our needs
    tokenized_user_query_len=len(tokenized_user_query)
    tokenized_user_query = np.array(tokenized_user_query).reshape(1, tokenized_user_query_len)
    ready_user_query = padding_doc(tokenized_user_query, max_length)

    #make prediction
    all_predictions = loaded_model.predict(ready_user_query)

    return all_predictions


def sort_predictions(all_predictions, classes):
    #as all_predictions is n dimentional array with shape (1,27)
    predictions = all_predictions[0]
    classes = np.array(classes)
    #use argsort to get sorted indices in dec order
    ids = np.argsort(-predictions)
    #sort classes as per sorted indices above
    classes = classes[ids]
    #sort predictions array in dec order
    predictions = -np.sort(-predictions)
    #get the intent with highest probability
    pred_intent = classes[0]

    return pred_intent


def ieee_faq_bot(user_query):
    #pass the user query through predictions function and get predicted intent
    all_predictions = predictions(user_query)
    predicted_intent = sort_predictions(all_predictions, unique_intent)

    return predicted_intent

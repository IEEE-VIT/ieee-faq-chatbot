import os
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
mongo_uri = os.getenv("mongo_uri")
client = MongoClient(mongo_uri)
db = client.get_database("ieee_faq_bot")
queries = db.user_input_queries

data = pd.DataFrame(list(queries.find()))
data.drop(data.columns[0], axis=1, inplace=True)
data = data.drop_duplicates(
    subset=["input_query"], keep='first', inplace=False, ignore_index=True)
data.dropna()
data.to_csv("new_user_data_raw.csv")

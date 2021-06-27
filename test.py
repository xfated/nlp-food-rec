from sentence_transformers import SentenceTransformer, util
from data_utils import *
import numpy as np
import pandas as pd
import torch
import faiss
import time

# Model details
model_name = 'msmarco-distilbert-base-v3'
version='v1'
model_save_path = 'output/review_emb-'+model_name+'-'+version

# model = SentenceTransformer(model_save_path)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Two lists of sentences
root = 'C:/Users/User/Documents/portfolio/food-review-scraper/reviewscraper/restaurant_data'
files = get_filepaths(root)

## Test with user input
restaurants = []
check_idx = 0
count = 0
rest_info = []
for idx, rest_path in enumerate(files):
    with open(rest_path,'r') as f:
        # Get data for this restaurant
        rest_data = json.load(f)
        if 'Singapore' not in rest_data['address']:
            continue
        #reviews = [f"{rest_data['name']} {review}" for review, score in rest_data['reviews'] if score >= 4]
        orig_tags = rest_data['review_tags']
        orig_name = rest_data['name']
        rest_info.append([orig_name, rest_data['address'], orig_tags, rest_data['summary']])
        restaurants.append(orig_name + ' ' + rest_data['address'] + ' ' + ' '.join(rest_data['review_tags'][:10]) + ' ' + rest_data['summary'])
        count += 1

rest_info_df = pd.DataFrame(rest_info, columns=['name','address','review_tags', 'summary'])

# Store embeddings
embeddings2 = model.encode(restaurants, convert_to_tensor=True)
rest_emb = embeddings2.cpu().detach().numpy().astype('float32')
index = faiss.IndexIDMap(faiss.IndexFlatIP(384))
index.add_with_ids(rest_emb, np.array(range(0,len(rest_emb))).astype('int64'))
faiss.write_index(index, 'rest_emb.index')

# Fetch restaurant info
def fetch_rest_info(idx):
    info = rest_info_df.iloc[idx]
    info_dict = dict()
    info_dict['name'] = info['name']
    info_dict['address'] = info['address']
    info_dict['review_tags'] = info['review_tags']
    info_dict['summary'] = info['summary']
    return info_dict

'''
query = query
top_k = number of results
index = faiss index
model = embedding model
'''
def search(query, top_k, index, model):
    t=time.time()
    query_vector = model.encode([query])
    top_k = index.search(query_vector, top_k)
    print('>>>> Results in Total Time: {}'.format(time.time()-t))
    top_k_ids = top_k[1].tolist()[0]
    top_k_ids = list(np.unique(top_k_ids))
    results =  [fetch_rest_info(idx) for idx in top_k_ids]
    return results

user_input = 'durian desserts'

# Best match
results = search(user_input, top_k=5, index=index, model=model)

print('\n')
for result in results:
    print(f"Name: {result['name']}, Tags: {result['review_tags']}\nSummary: {result['summary']}\n")   

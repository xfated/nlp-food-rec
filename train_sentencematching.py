from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime
import os
import gzip
import csv
import numpy as np
import json
from data_utils import *
import pandas as pd

### Print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

### Create dataset


### Specify training details
model_name = 'msmarco-distilbert-base-v3'
version='v1'
train_batch_size = 8
num_epochs = 4
model_save_path = 'output/review_emb-'+model_name+'-'+version

# Load a pre-trained sentence transformer model
model = SentenceTransformer(model_name)

## Prepare dataset for training
train_samples = []
test_samples = []
# Iterate through restaurant files and consolidate
# df_name = 'restaurant_train_data.csv'
# if os.path.exists(df_name):
#     logging.info("Preparing dataset from " + df_name)
#     df = pd.read_csv('restaurant_data.csv')
# else:
root = 'C:/Users/User/Documents/portfolio/food-review-scraper/reviewscraper/restaurant_data'
logging.info("Preparing dataset from " + root)
files = get_filepaths(root)
for rest_path in files:
    with open(rest_path,'r') as f:
        # Get data for this restaurant
        rest_data = json.load(f)
        if 'Singapore' not in rest_data['address']:
            continue
        reviews = [f"{rest_data['name']} {review}" for review, score in rest_data['reviews'] if score >= 4]
        orig_tags = rest_data['review_tags']
        orig_name = rest_data['name']

        # Get positive example            
        same_pairs = match_reviews_same(reviews, 10)
        # Get some negative example
        diff_pairs = match_reviews_diff(orig_name, reviews, orig_tags, files, max_restaurant=10, max_match=2)

        # print(len(reviews)) 
        # print(len(same_pairs))
        # print(len(diff_pairs))
        # exit()
        train_samples += same_pairs
        train_samples += diff_pairs
        

df = pd.DataFrame(train_samples, columns=['First', 'Second', 'Similarity'])
df = df.sample(frac=1, random_state=42)
test_split = 0.8
train_samples, test_samples = np.split(df, 
                       [int(test_split*len(df))])
train_samples, test_samples = train_samples.to_numpy(), test_samples.to_numpy()
logging.info("Data loaded")

# Convert to InputExample
train_samples = [InputExample(texts=[first, second], label=score) for first, second, score in train_samples]
test_samples = [InputExample(texts=[first, second], label=score) for first, second, score in test_samples]
logging.info("Converted to InputExample")

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)

# Development set: Measure correlation between cosine score and gold labels
logging.info("Read test dataset")
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='reviews-test')

# Configure the training.
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logging.info('Number of training examples: {}'.format(len(train_samples)))
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=2*len(test_samples), # 1000
          warmup_steps=warmup_steps,
          output_path=model_save_path)
          
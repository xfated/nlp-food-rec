from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util, InputExample
from sentence_transformers import models, datasets
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
# Iterate through restaurant files and consolidate
root = 'C:/Users/User/Documents/portfolio/food-review-scraper/reviewscraper/restaurant_data'
logging.info("Preparing dataset from " + root)
files = get_filepaths(root)
samples=[]
for idx, rest_path in enumerate(files):
    if idx % 100 == 0:
        logging.info("Done reading: {} files".format(idx))
    with open(rest_path,'r') as f:
        # Get data for this restaurant
        rest_data = json.load(f)
        if 'Singapore' not in rest_data['address']:
            continue
        reviews = [f"{review}" for review, score in rest_data['reviews'] if score >= 4]
        # orig_tags = rest_data['review_tags']
        orig_name = rest_data['name']

        pairs = get_keyword_review(orig_name, reviews, ngram_len=2, top_n=5)
        samples += pairs
        
        # for sample in samples:
        #     print(sample)
        #     print()
        # exit()

# split = 0.8
# train_split = split * len(samples)
# train_samples, test_samples = samples[:train_split], samples[train_split]
train_samples = samples
logging.info("Data loaded")
logging.info("Number of Training Examples:".format(len(train_samples)))
# logging.info("Number of Test Examples:".format(len(test_samples)))

# Convert to InputExample
train_samples = [InputExample(texts=[keywords, review]) for keywords, review in train_samples]
# test_samples = [InputExample(texts=[keywords, review]) for keywords, review in test_samples]
logging.info("Converted to InputExample")

train_dataloader = datasets.NoDuplicatesDataLoader(train_samples, batch_size=train_batch_size)
train_loss = losses.MultipleNegativesRankingLoss(model)

# Configure the training.
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logging.info('Number of training examples: {}'.format(len(train_samples)))
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          output_path=model_save_path)
          
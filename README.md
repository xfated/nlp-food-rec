## First model: v1
match_reviews_same: max_pairs = 10
match_reviews_diff: max_restaurant=5, max_match=5
train/test split:   0.8/0.2
batch_size:         8
epoch:              3
warmup_steps:       10%

## First model: v2
match_reviews_same: max_pairs = 5
match_reviews_diff: max_restaurant=5, max_match=5
train/test split:   0.8/0.2
batch_size:         8
epoch:              4
warmup_steps:       10%

## First model: v3
match_reviews_same: max_pairs = 10
match_reviews_diff: max_restaurant=10, max_match=3
train/test split:   0.8/0.2
batch_size:         8
epoch:              4
warmup_steps:       10%
tags: 50, multiply score by 2. clip to 1

## Second model: v1
Asymmetric search
model_name = msmarco-distilbert-base-v3
keyword review match: ngram_len=2, top_n=5
batch_size:         8
epoch:              4
warmup_steps:       10%




onnx-tf convert -i msmarco_distilbert_base_v4_onnx/msmarco_distilbert_base_v4.onnx -o msmarco-distilbert-v4


url:

rest_review_minilm
rest_review_distilbert
rest_review_distilbert_base (orig pretrained model)
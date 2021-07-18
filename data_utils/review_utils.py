import os
import json
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from proc_utils import filter_less_grams

def get_filepaths(root):
    for _root, dirs, files in os.walk(root):
        return [os.path.join(root, name) for name in files]

# Return pairs of string with label 1
# [review 1, review2, score]
def match_reviews_same(reviews, max=None, max_pairs=10):
    matches = []
    if max is None:
        max = len(reviews)
    for i in range(max):
        matches += [(reviews[i], second, 1) for second in reviews[i+1:min(i+1+max_pairs,max)]]
    return matches

# Return pairs of string with score
# Match reviews from different restaurant
# Score = num of same tag / total tag
def match_reviews_diff(cur_name, reviews, cur_tags, filepaths, max_restaurant=5, max_match=5):
    num_rest = len(filepaths)
    matches = []

    def intersection(lst1, lst2):
        return list(set(lst1) & set(lst2))

    # Match with X +other restaurants
    for _ in range(max_restaurant):
        # Search for a valid restaurant to match
        valid_rest = False
        while not valid_rest:
            # Select another restaurant at random
            idx = np.random.randint(num_rest)
            rest_path = filepaths[idx]
            with open(rest_path,'r') as f:
                rest_data = json.load(f)
            
            # Check if valid. Invalid if (non-singapore address) or (is same restaurant)
            if 'Singapore' not in rest_data['address'] or rest_data['name'] == cur_name:
                continue
            valid_rest = True
    
        # Valid restaurant found. Start matching based on tag.
        other_tags =  rest_data['review_tags']
        shorter = min(len(cur_tags), len(other_tags))
        common = intersection(cur_tags, other_tags)
        score = len(common) / shorter
        score = score * 2
        if score > 1:
            score = 1

        # Start matching up to max_match
        other_reviews = [f"{rest_data['name']} {review}" for review, score in rest_data['reviews'] if score >= 4]
        # Assert maxmatch condition to not exceed
        max_match = min(max_match, len(reviews), len(other_reviews))
        # Get max_match random from init reviews, match with 5 random sample from other restaurant
        random_cur_reviews = random.sample(reviews, max_match)
        for cur_review in random_cur_reviews:
            matches += [(cur_review, other, score) for other in random.sample(other_reviews, max_match)]
        
    return matches

def get_keyword_review(restaurant_name, reviews, ngram_len=3, top_n=5):
    tfIdf_vect = TfidfVectorizer(stop_words = 'english', ngram_range=(1,ngram_len), use_idf=True)
    tfIdf = tfIdf_vect.fit_transform(reviews)

    pairs = []
    for i in range(len(reviews)):
        # Get keywords in each review
        df = pd.DataFrame(tfIdf[i].T.todense(), index=tfIdf_vect.get_feature_names(),
                            columns=['tfidf'])
        df = df[df['tfidf'] > 0] # filter
        df = df.sort_values('tfidf', ascending=False)

        keywords = [(word, score) for word, score in zip(df.index.tolist(), df['tfidf'].values.tolist())]
        
        # Get top n probability first. Limit number of keywords for short review
        if len(reviews[i].split(' ')) < 50:
            top_n = 6
        keywords = keywords[:min(top_n, len(keywords))] # filter for top n results, or max num of keywords
        keywords = filter_less_grams(keywords, max_n=ngram_len) #get keywords in descending order
        keywords = ' '.join(keywords)

        # if i == 22:
        #     # print(df)
        #     print(keywords)
        #     print(reviews[i])
        #     exit()
        pairs.append((keywords, reviews[i]))
    
    pairs = [(keywords, f"{restaurant_name} {review}") for keywords, review in pairs]        
    return pairs, keywords    


if __name__ == "__main__":
    # root = 'C:/Users/User/Documents/portfolio/food-review-scraper/reviewscraper/restaurant_data'
    root = 'C:/Users/User/Documents/portfolio/MLPipeline/fastapi/restaurant_data'
    files = get_filepaths(root)

    max_len = 0
    longest = ''

    total_len = 0
    count = 0
    for idx, rest_path in enumerate(files):
        print(idx)
        with open(rest_path,'r') as f:
            rest_data = json.load(f)
            if 'Singapore' not in rest_data['address']:
                continue
            reviews = [f"{review}" for review, score in rest_data['reviews'] if score >= 4]
            orig_tags = rest_data['review_tags']
            orig_name = rest_data['name']            

            pairs = get_keyword_review(orig_name, reviews, ngram_len=2, top_n=10)
            if idx == 50:
                for pair in pairs:
                    print(pair)
                    print()
                exit()
    #         for review in reviews:
    #             sentence = review.split(' ')
    #             if len(sentence) > max_len:
    #                 max_len = len(sentence)
    #                 longest = review
    #             total_len += len(sentence)
    #             count += 1
                
    # print(max_len)
    # print(longest)
    # print('avg len:', total_len/count) 
           
            # Test pairs 
            # same_pairs = match_reviews_same(reviews, max_pairs=10)
            # diff_pairs = match_reviews_diff(orig_name, reviews, orig_tags, files)
            # print('num_reviews:',len(reviews))
            # print(len(same_pairs))
            # print(len(diff_pairs))
            # exit()
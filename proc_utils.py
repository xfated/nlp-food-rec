import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import os
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from collections import Counter
import re

'''
compute tfidf scores of each word in corpus
@Input args
texts list[string] corpus of texts

@Returns
words  List[string]
scores List[double]
'''
stopwords = set(stopwords.words('english'))
ignorewords = set(['i', 'the', 'one', 'this', 'go', 'always', 'it', 'we', 'people', 'rating', 'like'])

# Check if we want the word
def word_check(word):
    global stopwords
    global ignorewords
    
    if word in stopwords:
        return False
    if str.isdigit(word):
        return False
    if word in ignorewords:
        return False
    return True
    
def tfIdf(corpus):
    tfIdf_vect = TfidfVectorizer(use_idf=True)
    tfIdf = tfIdf_vect.fit_transform(corpus)
    
    # tfidf
    df = pd.DataFrame(tfIdf[0].T.todense(), 
        index=tfIdf_vect.get_feature_names(), columns=["tfidf"])
    # filter and sort
    df = df[df['tfidf'] > 0]
    df = df.sort_values('tfidf', ascending=False)

    words = df.index.tolist()
    scores = df['tfidf'].values.tolist()

    result = [[word,score] for word,score in zip(words, scores) if word_check(word)]

    return result

'''
Returns most common words that appear in > thres_ratio of reviews
'''
def common_words(corpus, thres_ratio=0.1):
    num_texts = len(corpus)

    # Extract words
    tokenizer = RegexpTokenizer(r'\w+')
    words = [str.lower(word) for text in corpus for word in tokenizer.tokenize(text) if word_check(str.lower(word))]
    word_count = Counter(words)
    results = [[word, count] for word, count in word_count.items() if count > num_texts*thres_ratio]
    return results

if __name__ == "__main__":
    root = 'C:/Users/User/Documents/portfolio/food-review-scraper/reviewscraper/restaurant_data'
    jsonfile = 'starbucks.json'
    filepath = os.path.join(root, jsonfile)
    with open(filepath,'r') as f:
        data = json.load(f)
    
    # Preproc
    print('Number of reviews:', len(data['reviews']))
    test_data = [review for review, rating in data['reviews']]

    ## Test tfidf 
    # test_data.append(data['about'])
    # results = tfIdf(test_data)
    # for res in results:
    #     print(res)
    results = common_words(test_data)
    print(results)
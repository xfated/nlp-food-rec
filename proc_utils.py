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
from nltk.util import ngrams
'''
compute tfidf scores of each word in corpus
@Input args
texts list[string] corpus of texts

@Returns
words  List[string]
scores List[double]
'''
stopwords = set(stopwords.words('english'))
ignorewords = set(['i', 'the', 'one', 'this', 'go', 'always', 'it', 'we', 'people', 'rating', 'like',
                    'got', 'end', 'would'])

# Check if we want the word
# Return False if we don't want
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

def ngram_check(ngram):
    for word in ngram:
        if not word_check(word): # if false, means fail
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
    tokenizer = RegexpTokenizer(r'[A-Za-z0-9]+')

    # unigram
    unigrams = [str.lower(word) for text in corpus for word in tokenizer.tokenize(text) if word_check(str.lower(word))]
    # bigram
    bigrams = [' '.join(bigram) for text in corpus for bigram in ngrams(tokenizer.tokenize(text),2) if ngram_check(bigram)]
    # trigrams
    trigrams = [' '.join(trigram) for text in corpus for trigram in ngrams(tokenizer.tokenize(text),3) if ngram_check(trigram)]

    words = unigrams + bigrams + trigrams
    word_count = Counter(words)
    results = [[word, count] for word, count in word_count.items() if count > num_texts*thres_ratio]
    return results

if __name__ == "__main__":
    root = 'C:/Users/User/Documents/portfolio/food-review-scraper/reviewscraper/restaurant_data'
    jsonfile = '63celsius__asia_square.json'
    filepath = os.path.join(root, jsonfile)
    with open(filepath,'r') as f:
        data = json.load(f)
    
    # Preproc
    print('Number of reviews:', len(data['reviews']))
    test_data = [review for review, rating in data['reviews']]

    ## Test tfidf 
    test_data.append(data['about'])
    results = tfIdf(test_data)
    print('tfidf:', results)

    ## Test common words
    results = common_words(test_data)
    print('common words:', results)
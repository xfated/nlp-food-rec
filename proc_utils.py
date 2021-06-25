import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
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
ignorewords = set()
ignorewords = set(['i', 'the', 'one', 'this', 'go', 'always', 'it', 'we', 'people', 'rating', 'like',
                    'got', 'end', 'would', 'note', 'closed', 'came', 'die', 'say', 'right'])

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

'''
NGRAM UTILS
'''
# Check if ngram is valid
def ngram_check(ngram):
    for word in ngram:
        if not word_check(str.lower(word)): # if false, means fail
            return False
    return True

def get_ngrams(corpus, n):
    tokenizer = RegexpTokenizer(r'[A-Za-z0-9]+')
    return [' '.join(ngram) for text in corpus for ngram in ngrams(tokenizer.tokenize(str.lower(text)),n) if ngram_check(ngram)]

# corpus = List[string]
# n = max n-gram
def get_ngrams_all(corpus, n):
    corpus_ngrams = []
    for i in range(1, n+1):
        corpus_ngrams += get_ngrams(corpus, i)
    return corpus_ngrams

# Filter
# if n-1 gram is found in n gram, should be omitted. Because likely has high score due to being found together in ngram
def filter_less_grams(results, max_n=3):
    filtered = []
    for i in range(1, max_n):
        words_ngram_prev = [word for word in results if len(word.split(' ')) == i] # get all (i)-gram
        words_ngram_cur = [word for word in results if len(word.split(' ')) == i + 1] # get all (i+1) gram

        # Check all smaller words, if is found in any larger word, don't add
        for word in words_ngram_prev:
            add = True
            for larger_word in words_ngram_cur:
                if word in larger_word:
                    add = False
                    break
            if add:
                filtered.append(word)
    filtered +=  [word for word in results if len(word.split(' ')) == max_n]
    return filtered

def tfIdf(corpus, ngram_len=2):
    tfIdf_vect = TfidfVectorizer(stop_words = 'english', ngram_range=(1,ngram_len), use_idf=True)
    tfIdf = tfIdf_vect.fit_transform(corpus)
    
    # tfidf
    df = pd.DataFrame(tfIdf[0].T.todense(), 
        index=tfIdf_vect.get_feature_names(), columns=["tfidf"])
    # filter and sort
    df = df[df['tfidf'] > 0]
    df = df.sort_values('tfidf', ascending=False)

    words = df.index.tolist()
    scores = df['tfidf'].values.tolist()

    result = [(word,score) for word,score in zip(words, scores) if ngram_check(word.split(' '))]

    return result

'''
Returns most common words that appear in > thres_ratio of reviews
'''
def countFreq(corpus, thres_ratio=0.1, ngram_len=3):
    num_texts = len(corpus)

    # apply vectorizer
    count_vect = CountVectorizer(stop_words = 'english', ngram_range=(1,ngram_len))
    count = count_vect.fit_transform(corpus)
    
    # get count
    words_sum = count.sum(axis=0)
    words_freq = [(word, words_sum[0, idx]) for word, idx in count_vect.vocabulary_.items() if words_sum[0, idx] > num_texts*thres_ratio]
    words_freq = [(word, count) for word, count in words_freq if ngram_check(word.split(' '))]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq

'''
Returns most common words that appear in > thres_ratio of reviews
Replaced with countFreq function
'''
def common_words(corpus, thres_ratio=0.1, ngram_len = 3):
    num_texts = len(corpus)

    # Extract words
    words = get_ngrams_all(corpus, ngram_len)
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

    # ## Test tfidf 
    test_data.append(data['about'])
    tfidf_results = tfIdf(test_data)
    # print('tfidf:', tfidf_results)

    # ## Test common words
    comm_results = common_words(test_data)
    print('common words:', comm_results)

    count = countFreq(test_data)
    print('count:', count)

    tfidf_terms = [text for text, score in tfidf_results]
    comm_terms = [text for text, score in comm_results]

    final = set(tfidf_terms + comm_terms)
    # print('Final:', final)

    filtered = filter_less_grams(final, 2)
    print()
    # print('filtered:', filtered)
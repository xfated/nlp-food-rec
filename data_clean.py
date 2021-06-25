from proc_utils import tfIdf, countFreq, filter_less_grams
import os
import numpy as np
import json

# Get tags
def get_tags(corpus, ngram_len=2):
    tfidf_results = tfIdf(corpus, ngram_len=ngram_len-1)
    # print('tfidf:', tfidf_results)

    # ## Test common words
    comm_results = countFreq(corpus, thres_ratio=0.15, ngram_len=ngram_len)
    # print('common words:', comm_results)

    tfidf_terms = [text for text, score in tfidf_results]
    comm_terms = [text for text, score in comm_results]

    final = set(tfidf_terms + comm_terms)
    # print('Final:', final)

    filtered = filter_less_grams(final, max_n=ngram_len)
    # print()
    # print('filtered:', filtered)
    return filtered

if __name__ == "__main__":

    for root, dirs, files in os.walk('C:/Users/User/Documents/portfolio/food-review-scraper/reviewscraper/restaurant_data'):
        num = np.random.randint(1000)
        print(num)
        for idx, name in enumerate(files):
            filepath = os.path.join(root, name)
        
            if idx == num or 'tai_cheong_bakery' in name:
                print('File:', name)
                with open(filepath,'r') as f:
                    data = json.load(f)
                
                # Preproc
                print('Number of reviews:', len(data['reviews']))
                test_data = [review for review, rating in data['reviews']]
                tags = get_tags(test_data)
                print(tags)
                # exit()
            
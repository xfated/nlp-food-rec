from proc_utils import tfIdf, countFreq, filter_less_grams
import os
import numpy as np
import json

# Get tags
def get_tags(corpus, ngram_len=3, use_comm=False):
    tfidf_results = tfIdf(corpus, ngram_len=ngram_len)
    # print('tfidf:', tfidf_results)

    # ## Test common words
    comm_results = countFreq(corpus, thres_ratio=0.15, ngram_len=ngram_len)
    # print('common words:', comm_results)

    tfidf_terms = [text for text, score in tfidf_results]
    comm_terms = [text for text, score in comm_results]

    final = tfidf_terms
    if use_comm:
        final = set(tfidf_terms + comm_terms)
    # print('Final:', final)

    filtered = filter_less_grams(final, max_n=ngram_len)
    # print()
    # print('filtered:', filtered)
    return filtered

if __name__ == "__main__":

    for root, dirs, files in os.walk('C:/Users/User/Documents/portfolio/food-review-scraper/reviewscraper/restaurant_data'):
        for idx, name in enumerate(files):
            # Get data
            filepath = os.path.join(root, name)
            with open(filepath,'r') as f:
                data = json.load(f)
            if 'Singapore' not in data['address']:
                continue

            # Get tags and save                
            try:
                # Extract tags
                rest_data = [review for review, rating in data['reviews']]
                rest_data.append(data['about'])
                tags = get_tags(rest_data)
                
                # Save
                data['review_tags'] = tags
                with open(filepath,'w') as json_file:
                    json.dump(data, json_file)
                print('Saved:', filepath)
            except Exception as e:
                print(e)
                continue
from proc_utils import tfIdf, countFreq, filter_less_grams
import os
import numpy as np
import json
import re

# Get tags
def get_tags(corpus, ngram_len=3, use_comm=False, top_n = 10):
    tfidf_results = tfIdf(corpus, ngram_len=ngram_len, top_n = top_n)
    # print('tfidf:', tfidf_results)

    # ## Test common words
    comm_results = countFreq(corpus, thres_ratio=0.15, ngram_len=ngram_len)
    # print('common words:', comm_results)

    tfidf_terms = [(text, score) for text, score in tfidf_results]
    comm_terms = [text for text, score in comm_results]

    final = tfidf_terms
    if use_comm:
        final = set(tfidf_terms + comm_terms)
    # print('Final:', final)

    # print(final)
    filtered = filter_less_grams(final, max_n=ngram_len)
    # print()
    # print('filtered:', filtered)
    return filtered

if __name__ == "__main__":
    '''
    Get relevant tags and add to restaurant json data
    '''
    for root, dirs, files in os.walk('C:/Users/User/Documents/portfolio/food-review-scraper/reviewscraper/restaurant_data'):
        for idx, name in enumerate(files):
            # Get data
            filepath = os.path.join(root, name)
            with open(filepath,'r') as f:
                data = json.load(f)
            if 'Singapore' not in data['address']:
                continue

            # Get tag                
            # Extract tags
            # rest_data = [review for review, rating in data['reviews']]
            # rest_data.append(data['about'])
            # tags = get_tags(rest_data, ngram_len=3, top_n=50)
            # data['review_tags'] = tags
            
            # Get region
            postal_code = re.findall("(Singapore) ([0-9]{6,6})", data["address"])
            if not postal_code:
                continue
            postal_code = postal_code[0][1][:2] # get first 2 digit
            postal_code = int(postal_code)
            region = ''
            if postal_code in (53, 54, 55, 82, # 19
                                56, 57, # 20
                                79, 80, # 28
                                ):
                region='North-East'
            if postal_code in (42, 43, 44, 45, # 15
                                46, 47, 48, # 16
                                49, 50, 81, # 17
                                51, 52, # 18
                                91
                                ):
                region='East'
            if postal_code in (69,70, 71, # 24
                                72, 73, # 25
                                75, 76 # 27
                                ):
                region='North'
            if postal_code in (1,2,3,4,5,6,  # 1
                              7, 8, # 2
                              14, 15, 16, # 3
                              9, 10, # 4
                              17, # 6
                              18, 19, # 7
                              20, 21, # 8
                              22, 23, # 9
                              24, 25, 26, 27, # 10
                              28, 29, 30, # 11
                              31, 32, 33, # 12
                              34, 35, 36, 37, # 13
                              38, 39, 40, 41, # 14
                              58, 59, # 21
                              77, 78 # 26
                            ):
                region='Central'
            if postal_code in (11, 12, 13,# 5
                                60, 61, 62, 63, 64, # 22
                                65, 66, 67, 68 # 23 
                                ):
                region='West'    
            if region == '':
                print(data['address'])
                print(postal_code)
                exit()
            data['region'] = region

            # Save
            with open(filepath,'w') as json_file:
                json.dump(data, json_file)
            print('Saved:', filepath)
            # exit()

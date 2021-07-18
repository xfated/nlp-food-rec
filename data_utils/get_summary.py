import re
import nltk
from sentence_transformers import SentenceTransformer, util
import numpy as np
from LexRank import degree_centrality_scores
from review_utils import *

# input: list of (review, score)
def get_summary(reviews, top_n):
    reviews = set([reviews for reviews,score in reviews])
    document = ' '.join(reviews)

    #Split the document into sentences
    sentences = nltk.sent_tokenize(document)
    # print("Num sentences:", len(sentences))

    #Compute the sentence embeddings
    embeddings = model.encode(sentences, convert_to_tensor=True)

    embeddings = embeddings.cpu()
    #Compute the pair-wise cosine similarities
    cos_scores = util.pytorch_cos_sim(embeddings, embeddings).numpy()

    #Compute the centrality for each sentence
    centrality_scores = degree_centrality_scores(cos_scores, threshold=0.1)

    #We argsort so that the first element is the sentence with the highest score
    most_central_sentence_indices = np.argsort(-centrality_scores)

    # Take top n sentences
    top_n = min(len(sentences), top_n)
    summary = ' '.join([sentences[idx].strip() for idx in most_central_sentence_indices[0:top_n]])
    return summary


if __name__ == "__main__":
    '''
    Use LexRank to extract most relevant sentences to use as summary.
    Add summary to restaurant json data
    '''
    model = SentenceTransformer('paraphrase-distilroberta-base-v1')

    # Get documents
    # root = 'C:/Users/User/Documents/portfolio/food-review-scraper/reviewscraper/restaurant_data'
    root = 'C:/Users/User/Documents/portfolio/MLPipeline/fastapi/restaurant_data'
    files = get_filepaths(root)

    # Get main sentences in review
    for idx, rest_path in enumerate(files):
        # print(rest_path)
        with open(rest_path,'r') as f:
            # Get data for this restaurant
            rest_data = json.load(f)
            if 'Singapore' not in rest_data['address']:
                continue

            ## Get summary
            # summary = get_summary(rest_data['reviews'], 10)
            # rest_data['summary'] = summary
            # print(summary)
            # print(len(summary.split(' ')))

            ## change url --> get default google url
            name = re.sub('[^a-zA-Z0-9 \n\.]', '', rest_data['name']).replace(" ", "+")
            address = rest_data["address"].replace(" ","+")
            url = f'https://google.com/search?q={name}+{address}'
            rest_data['url'] = url

            ## remove last Singapore 
            # address = rest_data['address']
            # split_address = address.split(' ')
            # if split_address[-1] == 'Singapore':
            #     split_address = split_address[:-1]
            #     address = ' '.join(split_address)
            #     rest_data['address'] = address
                
            with open(rest_path,'w') as json_file:
                json.dump(rest_data, json_file)
            print('Saved:', rest_path)
            # exit()
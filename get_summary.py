import re
import nltk
from sentence_transformers import SentenceTransformer, util
import numpy as np
from LexRank import degree_centrality_scores
from data_utils import *

# input: list of (review, score)
def get_summary(about, reviews, top_n):
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
    model = SentenceTransformer('paraphrase-distilroberta-base-v1')

    # Get documents
    root = 'C:/Users/User/Documents/portfolio/food-review-scraper/reviewscraper/restaurant_data'
    files = get_filepaths(root)

    # Get main sentences in review
    for idx, rest_path in enumerate(files):
        with open(rest_path,'r') as f:
            # Get data for this restaurant
            rest_data = json.load(f)
            if 'Singapore' not in rest_data['address']:
                continue

            summary = get_summary(rest_data['reviews'], 10)

            rest_data['summary'] = summary
            # print(summary)
            # print(len(summary.split(' ')))

            with open(rest_path,'w') as json_file:
                json.dump(rest_data, json_file)
            print('Saved:', rest_path)
            # exit()
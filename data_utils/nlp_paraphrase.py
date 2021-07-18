
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import os
import json
import torch

def get_filepaths(root):
    for _root, dirs, files in os.walk(root):
        return [os.path.join(root, name) for name in files]

''' Input 'text' ==> list of sentences '''
def paraphrase(model, tokenizer, text):
    # text = [' '.join([review1, review2]) for review1, review2 in zip(text[::2], text[1::2])]
    # print(text)
    batch = tokenizer(text,truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
    translated = model.generate(**batch,max_length=60,num_beams=4, num_return_sequences=1, temperature=1.5)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    summary = ' '.join(tgt_text)
    return summary

if __name__ == "__main__":
    model_name = 'tuner007/pegasus_paraphrase'
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

    
    # Get documents
    root = 'C:/Users/User/Documents/portfolio/MLPipeline/fastapi/restaurant_data'
    files = get_filepaths(root)

    # Get main sentences in review
    for idx, rest_path in enumerate(files):
        if idx != 100:
            continue
        with open(rest_path,'r') as f:
            # Get data for this restaurant
            rest_data = json.load(f)
            if 'Singapore' not in rest_data['address']:
                continue

            reviews_orig = rest_data['summary']
            reviews = [sentence.strip() + '.' for sentence in reviews_orig.split('.')]
            reviews = [review for review in reviews if len(review.split(' ')) > 2]
            # reviews_orig = rest_data['review_tags']
            # reviews = [' '.join(reviews_orig)]
            summary = paraphrase(model, tokenizer, reviews)
            
            print('orig summary: ---------')
            print(reviews_orig)
            print('final summary: ---------')
            print(summary)
            exit()

            # Add
            # rest_data['paraphrased_summary'] = summary
            # print(summary)
            # print(len(summary.split(' ')))

            # Save
            # with open(rest_path,'w') as json_file:
            #     json.dump(rest_data, json_file)
            # print('Saved:', rest_path)
            # exit()

''' With pipeline '''
# from transformers import pipeline
# summarizer = pipeline("summarization")

# ARTICLE = """ New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
# ... A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
# ... Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
# ... In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
# ... Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the
# ... 2010 marriage license application, according to court documents.
# ... Prosecutors said the marriages were part of an immigration scam.
# ... On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.
# ... After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective
# ... Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.
# ... All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.
# ... Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.
# ... Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.
# ... The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s
# ... Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
# ... Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.
# ... If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.
# ... """

# print(summarizer(ARTICLE, max_length=130, min_length=30, do_sample=False))

# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
import os
import json

def get_filepaths(root):
    for _root, dirs, files in os.walk(root):
        return [os.path.join(root, name) for name in files]


def generate_summary(model, tokenizer, text, max_length=512):
    inputs = tokenizer([text], return_tensors="pt", max_length=1024, truncation=True)
    # print('decode bef: ' + tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True, clean_up_tokenization_spaces=False))
    summary_ids = model.generate(inputs['input_ids'], max_length=max_length, min_length=max_length//2, num_beams=4, early_stopping=True)
    # print(outputs[0][:10])  
    summary = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
    return summary[0]

if __name__ == "__main__":
    '''
    Use LexRank to extract most relevant sentences to use as summary.
    Add summary to restaurant json data
    '''    
    # model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
    # tokenizer = AutoTokenizer.from_pretrained("t5-base")
    model = BartForConditionalGeneration.from_pretrained('sshleifer/distilbart-cnn-12-6') #'facebook/bart-large-cnn')
    tokenizer = BartTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6') #'facebook/bart-large-cnn')
    
    # Get documents
    root = 'C:/Users/User/Documents/portfolio/MLPipeline/fastapi/restaurant_data'
    files = get_filepaths(root)

    # Get main sentences in review
    for idx, rest_path in enumerate(files):
        with open(rest_path,'r') as f:
            # Get data for this restaurant
            rest_data = json.load(f)
            if 'Singapore' not in rest_data['address']:
                continue

            reviews = set([reviews for reviews,score in rest_data['reviews']])
            document = ' '.join(reviews)
            summary = generate_summary(model, tokenizer, document)
            
            # reviews = rest_data['summary']
            # summary = generate_summary(reviews, max_length=len(reviews.split(' ')))
            
            print('orig summary: ' + document)
            print('final summary:')
            print(len(summary.split(' ')))
            print(summary)
            exit()

            # Add
            # rest_data['summary'] = summary
            # print(summary)
            # print(len(summary.split(' ')))

            # Save
            # with open(rest_path,'w') as json_file:
            #     json.dump(rest_data, json_file)
            # print('Saved:', rest_path)
            # exit()
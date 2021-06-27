from os import truncate
import onnxruntime as rt
import numpy as np
from data_utils import *
import transformers

class review_emb():
    def __init__(self, model_path, tokenizer_path):
        # opt = rt.SessionOptions()
        self.sess = rt.InferenceSession(model_path)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    def get_emb(self, text):
        model_input = self.tokenizer.encode_plus(text, truncation=True, max_length=512)
        model_input = {name : np.int64(np.atleast_2d(value)) for name, value in model_input.items()}
        # print(model_input['input_ids'].shape)
        onnx_result = self.sess.run(None, model_input)

        emb = onnx_result[0][0]
        return emb


if __name__ == "__main__":
    # Load model
    model_path = 'rest_review_distilbert_wpool/rest_review_distilbert_wpool.onnx'
    # opt = rt.SessionOptions()
    # sess = rt.InferenceSession(model_path)

    # Get tokenizer
    model_name = 'msmarco-distilbert-base-v3'
    version='v1'
    model_save_path = 'output/review_emb-'+model_name+'-'+version
    # tokenizer = transformers.AutoTokenizer.from_pretrained(model_save_path, use_fast=True)

    text = "The food was very good and the service was sufficient. I like the crispy chicken wing and the rice with teriyaki sauce but a bit expensive to me. The sauce was simply delicious. The food came smothered in a sweet sticky sauce, this is obviously popular with some but certainly not us. The chicken very yummy and the price is reasonable.Highly reccomended Like both soya sauce and spicy versions. in the end i got a little more drumettes than wingettes so thank you kind lady :) 11/10 for her service I like the crispy chicken on there, they have taste salty and spicy, i like the salty ones, the spicy chicken are not too spicy, still can be eaten for whom dont like extreme spicy Last year we ate at 4 fingers and it was well cooked and really nice, so we went out of our way to try again. I expected it to be similar to KFC but it wasn't. Well cooked and fresh. I should of gone to KFC Well I know there's a lot of tasty fried chix, but this one here for me is addictive! Fries wise, to be honest there were a few instance I got fries that were not fresh, it is just normal, nothing impressive but good to have it as sides"
    text = text + ' ' + text

    # def get_emb(text):
    #     model_input = tokenizer.encode_plus(text)
    #     model_input = {name : np.int64(np.atleast_2d(value)) for name, value in model_input.items()}
    #     onnx_result = sess.run(None, model_input)

    #     emb = onnx_result[0][0]
    #     return emb

    model = review_emb(model_path, model_save_path)    
    res = model.get_emb(text)
    # print(res)

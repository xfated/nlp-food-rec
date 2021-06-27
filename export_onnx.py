from sentence_transformers import SentenceTransformer, util
from data_utils import *
import numpy as np
import pandas as pd
import time
from pathlib import Path

from transformers import convert_graph_to_onnx
import onnxruntime as rt
import transformers
import torch

import onnxruntime as rt
import numpy as np

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def export_model():
    # Model details
    model_name = 'msmarco-distilbert-base-v3'
    version='v1'
    model_save_path = 'output/review_emb-'+model_name+'-'+version

    # model = SentenceTransformer(model_save_path)
    model = SentenceTransformer(model_save_path, device='cuda')

    base_distilbert = convert_graph_to_onnx.load_graph_from_args("feature-extraction", "pt", model_save_path, None)
    with torch.no_grad():
        input_names, output_names, dynamic_axes, tokens = convert_graph_to_onnx.infer_shapes(base_distilbert, "pt")
        ordered_input_names, model_args = convert_graph_to_onnx.ensure_valid_input(base_distilbert.model, tokens, input_names)

    # input_names: ['input_ids', 'attention_mask']
    # output_names: ['output_0']
    # dynamic_axes: {'input_ids': {0: 'batch', 1: 'sequence'}, 'attention_mask': {0: 'batch', 1: 'sequence'}, 'output_0': {0: 'batch', 1: 'sequence'}}
    # tokens: {'input_ids': tensor([[ 101, 2023, 2003, 1037, 7099, 6434,  102]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1]])}
    # ordered_input_names: ['input_ids', 'attention_mask']

    torch.onnx.export(
        base_distilbert.model,
        model_args,
        f="rest_review_msmarco_distilbert.onnx",
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        use_external_data_format=False,
        enable_onnx_checker=True,
        opset_version=11,
    )

    ## Another method
    # from transformers.convert_graph_to_onnx import convert
    # convert(
    #     framework="pt", 
    #     model=model_save_path, 
    #     output=Path("rest_review_distilbert/rest_review_distilbert.onnx"), 
    #     opset=11
    # )




def test_onnx():
    model_name = 'msmarco-distilbert-base-v3'
    version='v1'
    model_save_path = 'output/review_emb-'+model_name+'-'+version
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_save_path, use_fast=True)
 
    sbert = transformers.FeatureExtractionPipeline(
        model=transformers.AutoModel.from_pretrained(model_save_path),
        tokenizer=transformers.AutoTokenizer.from_pretrained(model_save_path),
        framework="pt",
        device=0
        )

   
    opt = rt.SessionOptions()
    sess = rt.InferenceSession('rest_review_distilbert/rest_review_distilbert.onnx', opt)
    
    text = ""
    model_input = tokenizer.encode_plus(text)
    model_input = {name : np.int64(np.atleast_2d(value)) for name, value in model_input.items()}
    print(model_input)
    output = sess.run(None, model_input)

    assert np.allclose(
        np.array(sbert(text)[0]),
        output[0][0],
        atol=1e-4,
    )
    
if __name__ == "__main__":
    # export_model()
    test_onnx()

    # model_name = 'msmarco-distilbert-base-v3'
    # version='v1'
    # model_save_path = 'output/review_emb-'+model_name+'-'+version

    # model_pipeline = transformers.FeatureExtractionPipeline(
    #     model=transformers.AutoModel.from_pretrained(model_save_path),
    #     tokenizer=transformers.AutoTokenizer.from_pretrained(model_save_path, use_fast=True),
    #     framework="pt",
    #     device=-1
    # )


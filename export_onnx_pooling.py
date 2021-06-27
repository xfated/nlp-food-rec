from pathlib import Path

import onnx
import torch
import transformers
from pathlib import Path

import numpy as np
import onnxruntime as rt
from sentence_transformers import SentenceTransformer
from transformers import convert_graph_to_onnx

# Model location
model_name = 'msmarco-distilbert-base-v3'
version='v1'
model_save_path = 'output/review_emb-'+model_name+'-'+version

model_raw = SentenceTransformer(model_save_path, device="cuda")

### Loading Pipeline
model_pipeline = transformers.FeatureExtractionPipeline(
    model=transformers.AutoModel.from_pretrained(model_save_path),
    tokenizer=transformers.AutoTokenizer.from_pretrained(model_save_path, use_fast=True),
    framework="pt",
    device=-1
)

config = model_pipeline.model.config
tokenizer = model_pipeline.tokenizer

with torch.no_grad():
    input_names, output_names, dynamic_axes, tokens = convert_graph_to_onnx.infer_shapes(
        model_pipeline, 
        "pt"
    )
    ordered_input_names, model_args = convert_graph_to_onnx.ensure_valid_input(
        model_pipeline.model, tokens, input_names
    )

del dynamic_axes["output_0"] # Delete unused output
output_names = ["sentence_embedding"]
dynamic_axes["sentence_embedding"] = {0: 'batch'}

# Check that everything worked
print(output_names)
print(dynamic_axes)

class SentenceTransformer(transformers.DistilBertModel):
    def __init__(self, config):
        super().__init__(config)
        # Naming alias for ONNX output specification
        # Makes it easier to identify the layer
        self.sentence_embedding = torch.nn.Identity()
    
    def forward(self, input_ids, attention_mask):
        # Get the token embeddings from the base model
        token_embeddings = super().forward(
                input_ids, 
                attention_mask=attention_mask
            )[0]
        # Stack the pooling layer on top of it
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return self.sentence_embedding(sum_embeddings / sum_mask)

# Create the new model based on the config of the original pipeline
model = SentenceTransformer(config=config).from_pretrained(model_save_path)


span = 'i want durian puff'
assert np.allclose(
    model_raw.encode(span),
    model(**tokenizer(span, return_tensors="pt")).squeeze().detach().numpy(),
    atol=1e-6,
)

# Exporting model to onnx
model_name = 'rest_review_distilbert_wpool'
outdir = Path(model_name)
output = outdir / f"{model_name}.onnx"
outdir.mkdir(parents=True, exist_ok=True)

if output.exists():
    print(f"Model exists. Skipping creation")
else:
    print(f"Saving to {output}")
    # This is essentially a copy of transformers.convert_graph_to_onnx.convert
    torch.onnx.export(
        model,
        model_args,
        f=output.as_posix(),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        use_external_data_format=False,
        enable_onnx_checker=True,
        opset_version=12,
    )

# Checking model
onnx_model = onnx.load(output)
onnx.checker.check_model(onnx_model)
print('The model is checked!')

opt = rt.SessionOptions()
opt.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
opt.log_severity_level = 3
opt.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL

sess = rt.InferenceSession(str(output), opt) # Loads the model

model_input = tokenizer.encode_plus(span)
model_input = {name : np.int64(np.atleast_2d(value)) for name, value in model_input.items()}
onnx_result = sess.run(None, model_input)

print(np.array(onnx_result).shape)
assert np.allclose(model_raw.encode(span), onnx_result, atol=1e-6)
assert np.allclose(
    model(**tokenizer(span, return_tensors="pt")).squeeze().detach().numpy(), 
    onnx_result, 
    atol=1e-6
)
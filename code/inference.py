
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
from optimum.onnxruntime import ORTModelForFeatureExtraction

model_name = 'model' # This has to be the same as the one inside onnx_path

# Helper: Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
    return torch.sum(model_output * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def model_fn(model_dir):
    # load tokenizer and neuron model from model_dir
    model = ORTModelForFeatureExtraction.from_pretrained(model_dir, file_name=f"{model_name}.onnx")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    return model, tokenizer


def predict_fn(data, model_tokenizer_model_config):
    # destruct model and tokenizer
    model, tokenizer = model_tokenizer_model_config

    # Tokenize sentences
    inputs = data.pop("inputs", data)
    encoded_inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors='pt')
    
    # Compute token embeddings
    with torch.no_grad():
        model_outputs = model(**encoded_inputs)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_outputs["last_hidden_state"], encoded_inputs['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    # return dictonary, which will be json serializable
    return {"vectors": sentence_embeddings.tolist()}


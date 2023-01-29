from transformers import RobertaTokenizer, RobertaModel
from transformers import AutoTokenizer, AutoModel
import torch


def embed_text(model, tokenizer, text):
    with torch.no_grad():
        encoded_input = tokenizer(text, return_tensors='pt').to("cuda")
        output = model(**encoded_input)['last_hidden_state'][0]
    return torch.mean(output, dim=0)

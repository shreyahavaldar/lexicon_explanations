from transformers import RobertaTokenizer, RobertaModel
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import torch


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / \
        torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def embed_text(model, tokenizer, text):
    with torch.no_grad():
        encoded_input = tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors='pt').to("cuda")
        output = model(**encoded_input)
    embedding = mean_pooling(output, encoded_input['attention_mask'])
    embedding = F.normalize(embedding, p=2, dim=1)
    return embedding

from transformers import RobertaTokenizer, RobertaModel
from transformers import AutoTokenizer, AutoModel
import torch


def embed_text(text):
    # tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    # model = RobertaModel.from_pretrained('roberta-large')
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    model = AutoModel.from_pretrained("roberta-large")

    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)['last_hidden_state'][0]
    return torch.mean(output, dim=0)

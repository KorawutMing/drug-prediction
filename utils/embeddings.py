from FlagEmbedding import BGEM3FlagModel
from transformers import AutoTokenizer, BertModel
import torch

BGE_M3_MODEL = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False, device="cuda")
BERT_TOKENIZER = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
BERT_MODEL = BertModel.from_pretrained("google-bert/bert-base-uncased")

def genEmbsBge(texts: list):
    embs = BGE_M3_MODEL.encode(
        texts, return_dense=True, return_sparse=True, return_colbert_vecs=False)
    return embs['dense_vecs'], embs['lexical_weights']

def genEmbsBert(texts: list, padding: bool):
    inputs = BERT_TOKENIZER(texts, return_tensors="pt", padding = padding)
    outputs = BERT_MODEL(**inputs)
    return outputs.last_hidden_state
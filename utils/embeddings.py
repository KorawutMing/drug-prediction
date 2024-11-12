from FlagEmbedding import BGEM3FlagModel

MODEL = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

def genEmbs(texts: list):
    embs = MODEL.encode(
        texts, return_dense=True, return_sparse=True, return_colbert_vecs=False)
    return embs['dense_vecs'], embs['lexical_weights']
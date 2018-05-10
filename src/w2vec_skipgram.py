#W2vec on Gensim
import gensim

import collections
import os
import zipfile
import string

from nltk.corpus import stopwords
import numpy as np 
import tensorflow as tf

from src.word_embedding_utils_v2 import clean_corpus, build_w2id_dict, tokenize_text_data, corpus_to_idx




# #data is stored in numpy arrays [[post1], [post2], ] 
def read_data(file_name = 'dev_set_text'):
    data = np.load(file_name)
    stops = stopwords.words('english')
    cleaned_corpus = clean_corpus(data, stops)
    return(cleaned_corpus)

# def build_w2vec(data, w2id, window_size = 3, vocab_size=len(w2id), vector_dim = 300, epochs = 100000):
#     print_header("Creating the w2vec architecture: ")



if __name__ == "__main__":
    clean_corpus  = read_data(file_name = 'dev_set_text')
    model_w2v = gensim.models.Word2Vec(clean_corpus, sg=1,  size = 150, window=5, min_count=10)
    model_w2v.train(clean_corpus, total_examples = len(clean_corpus), epochs=100)
    model_w2v.save("w2vec_genism")


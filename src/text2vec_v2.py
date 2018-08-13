"""
Doc2Vec Model:

Compute and fit a Doc2Vec model to obtain 
document vectors. 

From there you can split documents and use
vectors to do classification
"""

import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy as np
import random
import os
import pickle 
import string
import requests 
import collections
import io 
import tarfile
import word_embedding_utils_v2
from nltk.corpus import stopwords
from tensorflow.python.framework import ops

def run_text2v(filepath='/Users/dentonzhao/Projects/StackExchange-Analysis/dataframes/dev_set_df', batch_size=500,
            vocabulary_size = 25000, generations = 100000,
            model_learning_rate = 0.001, embedding_size = 200, 
            doc_embedding_size=100, window_size=3, strategy="doc2vec"):

    ops.reset_default_graph()
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    data_folder_name = 'temp_models'
    if not os.path.exists(data_folder_name):
        os.makedirs(data_folder_name)

    sess = tf.Session()

    ################################################################
    #Model Parameters
    ################################################################
    batch_size = 500


    #Will aim for total embedding size of 300
    embedding_size = 200
    doc_embedding_size = 100
    concatenated_size = embedding_size + doc_embedding_size

    #Number of negative examples to sample
    num_sampled = int(batch_size/2) 
    window_size = 3

    #Add checkpoints to training
    save_embeddings_every = 5000
    print_valid_every = 5000
    print_loss_every =100

    stops = stopwords.words('english')
    # stops = []

    ################################################################


    #Now load in the data#
    print("Loading Data")
    # fp="/Users/dentonzhao/Projects/StackExchange-Analysis/clean_dev_set.csv/part-00000-4df5cbe6-3d8e-44ad-a26c-42f2303b93a0-c000.csv"
    corpus = word_embedding_utils_v2.get_text_from_pickle(filepath)

    #clean the loaded data
    print("Cleaning corpus")
    cleaned_corpus = word_embedding_utils_v2.clean_corpus(corpus, stops)

    # post in corpus must contain at least 3 words
    # target = [target[ix] for ix, x in enumerate(cleaned_corpus) if len(x.split()) > window_size]
    cleaned_corpus = [x for x in cleaned_corpus if len(x) > window_size]    
    # assert(len(target)==len(texts))




    #build dataset
    print("Constructing w2id")
    w2id = word_embedding_utils_v2.build_w2id_dict(cleaned_corpus, vocabulary_size)
    idx2word = dict(zip(w2id.values(), w2id.keys()))
    print("obtaining data_stream")
    data_stream = word_embedding_utils_v2.corpus_to_idx(cleaned_corpus, w2id)



    # We pick a few(6) test word indices for validation.
    validation_idx = np.random.randint(1, vocabulary_size, size=6)
    # Later we will have to transform these into indices

    print('Creating Model')
    ###########################################
    #define the embeddings:
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size],-1.0, 1.0),name="embeddings")
    doc_embeddings = tf.Variable(tf.random_uniform([len(data_stream), doc_embedding_size], 
                                                    minval=-1.0, 
                                                    maxval=1.0), name="doc_embeddings")

    # NCE loss parameters
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, concatenated_size],
                                                stddev=1.0 / np.sqrt(concatenated_size)), name="nce_weights")
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]), name="nce_biases")

    # Create data/target placeholders
    x_inputs = tf.placeholder(tf.int32, shape=[None, window_size + 1]) # plus 1 for doc index
    y_target = tf.placeholder(tf.int32, shape=[None, 1])
    valid_dataset = tf.constant(validation_idx, dtype=tf.int32)

    # Lookup the word embedding
    # Add together element embeddings in window:
    embed = tf.zeros([batch_size, embedding_size])

    #add together elements because for a given target, there are window_size inputs
    #this embed will now be an estimated vector for y?
    for element in range(window_size):
        embed += tf.nn.embedding_lookup(embeddings, x_inputs[:, element])

    #get the doc_indices
    doc_indices = tf.slice(x_inputs, [0,window_size],[batch_size,1])
    #get the doc embedding
    doc_embed = tf.nn.embedding_lookup(doc_embeddings,doc_indices)

    # concatenate embeddings
    final_embed = tf.concat([embed, tf.squeeze(doc_embed)],1)

    # Get loss from prediction
    loss = tf.reduce_mean(tf.nn.nce_loss(weights = nce_weights,
                                        biases = nce_biases, 
                                        inputs=final_embed, 
                                        labels = y_target,
                                        num_sampled = num_sampled,
                                        num_classes = vocabulary_size))
                                        
    # Create optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=model_learning_rate)
    train_step = optimizer.minimize(loss)

    # Cosine similarity between words
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    # Create model saving operation
    saver = tf.train.Saver({"embeddings": embeddings, "doc_embeddings": doc_embeddings})

    #Add variable initializer.
    init = tf.initialize_all_variables()
    sess.run(init)

    # Run the model.
    print('Starting Training')
    loss_vec = []
    loss_x_vec = []
    for i in range(generations):
        batch_inputs, batch_labels = word_embedding_utils_v2.generate_batch_data(data_stream, batch_size,
                                                                    window_size, method=strategy)
        feed_dict = {x_inputs : batch_inputs, y_target : batch_labels}

        # Run the train step
        sess.run(train_step, feed_dict=feed_dict)

        # Return the loss
        if (i+1) % print_loss_every == 0:
            loss_val = sess.run(loss, feed_dict=feed_dict)
            loss_vec.append(loss_val)
            loss_x_vec.append(i+1)
            print('Loss at step {} : {}'.format(i+1, loss_val))
        
        # Validation: Print some random words and top 5 related words
        if (i+1) % print_valid_every == 0:
            sim = sess.run(similarity, feed_dict=feed_dict)
            for j in range(len(validation_idx)):
                valid_word = idx2word[validation_idx[j]]
                top_k = 5 # number of nearest neighbors
                nearest = (-sim[j, :]).argsort()[1:top_k+1]
                log_str = "Nearest to {}:".format(valid_word)
                for k in range(top_k):
                    close_word = idx2word[nearest[k]]
                    log_str = '{} {},'.format(log_str, close_word)
                print(log_str)
                
        # Save dictionary + embeddings
        if (i+1) % save_embeddings_every == 0:
            # Save vocabulary dictionary
            with open(os.path.join(data_folder_name,'vocab.pkl'), 'wb') as f:
                pickle.dump(w2id, f)
            
            # Save embeddings
            model_checkpoint_path = os.path.join(os.getcwd(),data_folder_name,'text2vec_vocab_embeddings.ckpt')
            save_path = saver.save(sess, model_checkpoint_path)
            print('Model saved in file: {}'.format(save_path))

if __name__=='__main__':
    run_text2v(filepath='/Users/dentonzhao/Projects/StackExchange-Analysis/dataframes/dev_set_df', batch_size=500,
            vocabulary_size = 25000, generations = 100000,
            model_learning_rate = 0.001, embedding_size = 200, 
            doc_embedding_size=100, window_size=3, strategy="doc2vec")
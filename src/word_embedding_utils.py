import string
import os
import io
import tarfile 
import collections
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords

def cleanhtml_test(raw_html):
    """
    Removes the code stuff and the html tags inside a post
    """
    code = re.compile('<pre><code>.*?</code></pre>')
    sans_code = re.sub(code, ' ', raw_html)
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', sans_code)
    return cleantext

def get_text_data_test(fp):
    """
    Combines the title and body of a file after passing it through htmlcleaner
    Now you will have a list of posts.
    """
    df = pd.read_csv(fp, header=None)
    df.columns = ['postid', 'title', 'body' ]
    corpus = df["title"].apply(cleanhtml_test) + " "+ df["body"].apply(cleanhtml_test)
    return corpus

def clean_corpus(corpus, stops):
    """
    Inside our post data, lets remove the punctuations, 
    lowercase the words and represent each post as a series of "cleaned words"
    """
    pt = string.punctuation
    cleaned_corpus =[]
    for post in corpus:
        cleaned_post =[]
        tokens = post.split(" ")

        #remove any single word punctuations
        tokens = list(filter(lambda token: token not in pt, tokens))
        #no we have a list of words for each post. go into each word and clean it!
        
        #lowercase
        cleaned_tokens = [token.lower() for token in tokens]
        #remove punctuation:
        cleaned_tokens = [''.join(letter for letter in token if letter not in pt) for token in cleaned_tokens]
        #remove numbers
        cleaned_tokens = [''.join(letter for letter in token if letter not in '0123456789') for token in cleaned_tokens]

        # remove stopwords
        cleaned_tokens = list(filter(lambda token: token not in stops, cleaned_tokens))
        # remove empty elements:
        cleaned_tokens = list(filter(None, cleaned_tokens))

        cleaned_corpus.append(cleaned_tokens)

    return(cleaned_corpus)

def tokenize_text_data(cleaned_corpus):
    tokens = []
    for post in cleaned_corpus:
        for token in post:
            tokens.append(token)
    return(tokens)


def build_w2id_dict(cleaned_corpus, vocabulary_size):
    """
    For a given list of words and vocabulary size, make:
        1. data_stream - a stream of word_ids in order of appearence in corpus
        2. count - count of unique words of shape: (vocabulary_size,2)
        3. w2id - lookup dictionary maping a word to its "id"
        4. id2word - lookup dictionary mapping an id to word. (reverse of w2id) 
    """
    
    # Initialize list of [word, word_count] for each word, starting with unknown
    # Need to turn cleaned_corpus(list of posts) into list of words
    count = [['UNK', -1]]

    # tokens = tokenize_text_data(clean_corpus)
    tokens = []
    for post in cleaned_corpus:
        for token in post:
            tokens.append(token)
    
    # Now add most frequent words to the count. 
    count.extend(collections.Counter(tokens).most_common(vocabulary_size-1))
    
    # Now create the w2id dictionary. This be {word: unique_id} up to our word limit
    w2id = {}
    for word, _ in count:
        w2id[word] = len(w2id)
    return(w2id)
    # Now map word in corpus from word to unique_id
    
def corpus_to_idx(clean_corpus, w2id):
    """
    From clean_corpus and w2id_dict map words to index
    
    """    
    data_stream = []
    unk_count = 0
    
    for post in clean_corpus:
        post_idx_data = []
        for word in post:
        #get the index of the word 
        #if cant find in the w2id dict, treat as UNK
            word_ix = w2id.get(word,0)
            post_idx_data.append(word_ix)
        data_stream.append(post_idx_data)

    # count[0][1] = unk_count
    # id2word = dict(zip(w2id.values(), w2id.keys()))
    # del tokens
    # print out stats
    # print('most common words (+UNK):', count[:10])
    # print('sample data:', w2id[:10], [id2word[i] for i in data_stream[:10]])
    # return (data_stream, count, w2id, id2word)
    return(data_stream)

#Function to generate a training batch for the skip-gram model.

def generate_batch_data(data_stream, batch_size, window_size, method ='skip_gram'):
    #populate batch_data(the "target" word we are given)
    batch_data = []
    #populate the label_data(the "target" word we are predicting)
    label_data = []

    while len(batch_data) < batch_size:
        #pick a random document in the corpus
        rand_document_ix = int(np.random.choice(len(data_stream), size=1))
        rand_document = data_stream[rand_document_ix]

        #generate the windows to look at: [nwords behind, target, nwords ahead]
        window_sequences = []
        #generate the indices of the target word for each window
        label_indices=[]

        for window_num, _ in enumerate(rand_document):
            window = [rand_document[max((window_num - window_size),0): (window_num + window_size + 1)]] 
            window_sequences.append(window)
        
        for ix, _ in enumerate(window_sequences):
            if ix < window_size:
                label_index = ix
            else:
                #target will always be on same index as window_size once window is fully populated
                label_index = window_size
            label_indices.append(label_index)

        if method == 'skip_gram':
            batches_and_labels = []
            for seq, label_index in zip(window_sequences):
                batches_and_labels.append(
                                        (seq[label_index], 
                                        seq[:label_index] + seq[(label_index+1):])
                                        )
            #batches_and_labels in form like this: [(batch, ['labels])]
            #convert now to tuple key pairs
            tuple_data = []
            for batch, labels in batches_and_labels:
                for label in labels:
                    tuple_data.append(batch, label)
            batch, labels = [list(words) for words in zip(*tuple_data)]
        
        elif method == 'doc2vec':
            # For doc2vec we keep LHS window only to predict target word
            # [nwords_behind, target]
            batches_and_labels = []
            for i in range(0, len(rand_document)-window_size):
                target = rand_document[i+window_size]
                n_words_behind = rand_document[i:i+window_size]
                batches_and_labels.append((n_words_behind, target))
            batch, labels = [list(x) for x in zip(*batches_and_labels)]
            # Add document index to batch!! Remember that we must extract the last index in batch for the doc-index
            batch = [x + [rand_document_ix] for x in batch]
        else:
            raise ValueError('Only "skip_gram", "doc2vec" are valid inputs')

        batch_data.extend(batch[:batch_size])
        label_data.extend(labels[:batch_size])
    #previously we just threw everything into the batch, to ensure we clipped everything correctly,
    #lets just clip the batches
    batch_data = batch_data[:batch_size]
    label_data = label_data[:batch_size]

    #convert to numpy array to talk with tf.
    batch_data = np.array(batch_data)
    label_data = np.transpose(np.array([label_data]))

    return(batch_data, label_data)




# fp="/Users/dentonzhao/Projects/StackExchange-Analysis/clean_dev_set.csv/part-00000-4df5cbe6-3d8e-44ad-a26c-42f2303b93a0-c000.csv"
# text_data = get_text_data_test(fp)
# # words = tokenize_text_data(text_data)
# stop_words = set(stopwords.words('english'))
# cleaned_text_data = clean_text_data(text_data, stop_words)
# tokens = tokenize_text_data(cleaned_text_data)
# data_stream, count, w2id, id2word = build_embedding_lookups(tokens, 5000)



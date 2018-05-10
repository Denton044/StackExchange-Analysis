import gensim
import numpy as np 
from nltk.corpus import stopwords
from src.word_embedding_utils_v2 import clean_corpus, build_w2id_dict, tokenize_text_data
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model, Sequential
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score




#### preprocessing
def read_data(file_name = 'dev_set_text'):
    data = np.load(file_name)
    stops = stopwords.words('english')
    cleaned_corpus = clean_corpus(data, stops)
    return(cleaned_corpus)

def get_wv_matrix(genism_model):
    #build the np matrix
    embedding_shape = (len(genism_model.wv.vocab), genism_model.trainables.layer1_size)
    embedding_matrix = np.zeros(embedding_shape)

    #insert the data from model:
    for index in range(len(genism_model.wv.vocab)):
        embedding_vector = genism_model.wv[genism_model.wv.index2word[index]]
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
    return(embedding_matrix)

def corpus_to_idx(clean_corpus, genism_model):
    """
    From clean_corpus and w2id_dict map words to index
    
    """    
    vocabulary = genism_model.wv.vocab
    for post in clean_corpus:
        for index, word in enumerate(post):
            if word in vocabulary:
                post[index] = vocabulary[word].index
            else:
                #we will then put all unknowns into an "0" index, represented by len(vocab) + 1
                post[index] = 0
    return(clean_corpus)

def pickle_wordid_corpus(file_names=['dev_set_text', 'hold_set_text'], model = 'w2vec_genism'):
    w2v_model = gensim.models.Word2Vec.load(model)
    for item in file_names:
        print('cleaning file: ', item)
        clean_corpus = read_data(file_name = item)
        print('converting')
        word_ids = corpus_to_idx(clean_corpus, w2v_model)
        np.save(item + "_ids", word_ids)


####


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



def run_CNN(items_pickled=True, words_to_consider = 100):
    ###run a CNN with 100 words to consider###
    
    while items_pickled == True:
        dev_set_ids = np.load('dev_set_text_ids.npy')
        dev_set_labels = np.load('dev_set_labels')

        w2vec_model = gensim.models.Word2Vec.load('w2vec_genism')
        vocabulary = w2vec_model.wv.vocab
        #split into train, test split
        
        #pad data 
        data=pad_sequences(dev_set_ids, maxlen=words_to_consider, padding='post', truncating='post', value=0)

        x_train, x_validate, y_train, y_validate = train_test_split(data, dev_set_labels, 
                                                                    test_size=0.2, random_state=42)

        #make the embeddings
        embedding_matrix = get_wv_matrix(w2vec_model)

        embeddings = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1],
                            weights=[embedding_matrix], input_length = words_to_consider)

        #great now lets run the CNN
        cnn_test = Sequential()

        #first conv layer + max pool
        cnn_test.add(embeddings)
        cnn_test.add(Conv1D(filters=100, kernel_size = 5, activation = 'relu', strides = 1))
        cnn_test.add(MaxPooling1D(5))

        #2nd conv layer + max pooling
        cnn_test.add(Conv1D(filters =100, kernel_size = 2, activation='relu'))
        cnn_test.add(MaxPooling1D(5))

        #flatten and then connect
        cnn_test.add(Flatten())
        cnn_test.add(Dense(256, activation = 'relu'))

        #output layer with sigmoid activation
        cnn_test.add(Dense(y_train.shape[1], activation = 'sigmoid'))

        # Compile settings
        print('\tcompiler settings complete!')
        cnn_test.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = [f1])

        cnn_test.fit(x_train, y_train, validation_data=(x_validate, y_validate), epochs = 5, batch_size= 1000, verbose=2)
        cnn_test.save('cnn_model.h5')
    else:
        pickle_wordid_corpus()
        items_pickled=True
        run_CNN()



if __name__ == "__main__":
    run_CNN()
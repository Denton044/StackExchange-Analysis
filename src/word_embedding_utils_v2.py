import string
import collections
import numpy as np
import re
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
from nltk.corpus import stopwords


def get_data(post_filepath ='/Users/dentonzhao/Downloads/Questions.csv', 
             tag_filepath = '/Users/dentonzhao/Downloads/Tags.csv'):
    """
    Load in raw data from post and file df, merge two
    INPUTS:
        post_filepath: string of filepath of posts csv
        tag_filepath: string of the filepath of the tag csv
    OUTPUT:
        dataframe with inner merge of the two dataframes
    """
    posts = pd.read_csv(post_filepath, encoding = 'latin-1')
    tags = pd.read_csv(tag_filepath, encoding = 'latin-1')
    #merge the two dataframes
    all_data = pd.merge(posts, tags, how = 'inner', on = 'Id')
    #only interested in these columns
    all_data['Text'] = all_data['Title'] + " " + all_data['Body']
    all_data = all_data[['Id', 'Text', 'Tag']]
    return(all_data)

def get_topn_tags(dataframe, n_tags=15):
    top_tags = list(dataframe['Tag'].value_counts()[:n_tags].index)
    relevant_posts = dataframe[dataframe['Tag'].isin(top_tags)].reset_index()
    relevant_posts = relevant_posts.groupby(['Id','Text'])['Tag'].apply(list).reset_index()
    relevant_posts.to_pickle('relevant_post_pkl')

# def clean_html_tags1(raw_html):
#     """
#     Removes the code stuff and the html tags inside a post
#     """
#     code = re.compile('<pre><code>.*?</code></pre>')
#     sans_code = re.sub(code, ' ', raw_html)
#     cleanr = re.compile('<.*?>')
#     cleantext = re.sub(cleanr, ' ', sans_code)
#     return (cleantext)

def clean_html_tags(raw_text):
    soup = BeautifulSoup(raw_text, 'html.parser')
    [s.extract() for s in soup('code')]
    return(str(soup.get_text()))

def remove_white_spaces(text):
    new_text = text.replace('"','').replace("\n","").replace("\t","")
    return(new_text)


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

    return cleaned_corpus

def clean_and_split(filepath='relevant_post_pkl', hold_set_percent = 0.2):
    """Given data, clean the text and split off the data into devset and holdoutset"""
    #clean this data
    #cleaning the X_data
    print("removing html tags")
    relevant_posts = pd.read_pickle(filepath)
    relevant_posts['Text'] = relevant_posts['Text'].apply(clean_html_tags).str.lower()
    relevant_posts['Text'] = relevant_posts['Text'].apply(remove_white_spaces)
    X = relevant_posts['Text'].values
    
    # print('tokenizing the corpus')
    # stops = stopwords.words('english')
    # X = clean_corpus(X, stops)

    #clean the y_data:
    print("Cleaning Labels")
    y = relevant_posts['Tag'].values
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(y)

    label_dict = {}
    for index, label in enumerate(mlb.classes_):
        label_dict[index] = label
    
    # split the data:
    print("Split and writing to csvs")
    X_dev, X_hold, Y_dev, Y_hold = train_test_split(X, Y, test_size = hold_set_percent, random_state = 42)
    X_dev.dump('dev_set_text')
    X_hold.dump('hold_set_text')
    Y_dev.dump('dev_set_labels')
    Y_hold.dump('hold_set_labels')
    return(label_dict)

def build_w2id_dict(clean_corpus, vocabulary_size):
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
    for post in clean_corpus:
        for token in post:
            tokens.append(token)
    
    # Now add most frequent words to the count. 
    count.extend(collections.Counter(tokens).most_common(vocabulary_size-1))
    
    # Now create the w2id dictionary. This be {word: unique_id} up to our word limit
    w2id = {}
    for word, _ in count:
        w2id[word] = len(w2id)
    return(w2id)

def tokenize_text_data(cleaned_corpus):
    tokens = []
    for post in cleaned_corpus:
        for token in post:
            tokens.append(token)
    return(tokens)




#     # Now map word in corpus from word to unique_id
    
def corpus_to_idx(clean_corpus, w2id):
    """
    From clean_corpus and w2id_dict map words to index
    
    """    
    for post in clean_corpus:
        for index, word in enumerate(post):
        #get the index of the word 
        #if cant find in the w2id dict, treat as UNK
            post[index] = w2id.get(word,0)
        
    # count[0][1] = unk_count
    # id2word = dict(zip(w2id.values(), w2id.keys()))
    # del tokens
    # print out stats
    # print('most common words (+UNK):', count[:10])
    # print('sample data:', w2id[:10], [id2word[i] for i in data_stream[:10]])
    # return (data_stream, count, w2id, id2word)
    return(clean_corpus)

# if __name__ == "__main__":
#     data = clean_and_split()
#     print

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






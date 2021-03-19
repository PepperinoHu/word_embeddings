#
# SI630 Homework 2: word2vec
#
# GENERAL NOTES:
#
# We've provided skeleton code for the basic functions of word2vec, which will
# let you get started. The code tasks are
#
#  (1) preprocessing the text input to covert it into a sequence of token IDs
#  (2) writing the core training loop that uses those IDs as training examples
#  (3) writing the gradient desent part that update the weights based on how
#      correct the model's predictions were
#
# Of the three parts, Part 3 is the trickiest to get right and where you're
# likely to spend most of your time. You can do a simple job on Part 1 initally
# to get the model up and running so you can start on Part 2 and then begin
# implementing Part 3
#
# Word2vec itself is a complex set of software and we're only implementing (1)
# negative sampling, (2) rare word deletion, and (3) frequent word subsampling
# in this. You may see more advanced/complex methods desribed elsewhere.
#
# Note that the model implemented in this homework will take time to fully
# train. However, you can definitely still get intermediate results that confirm
# your model is working by running for fewer steps (or epochs). The easiest way
# to test is to look at the neighest neighbors for common words. We've built
# this in for you already so that after training, the code will print out the
# nearest neighbors of a few works. Usually after ~100K steps, the neighest
# neighbors for at least some of the words will start to make sense.
#
# Other helpful debugging notes:
#
# 1. While you get started, you can turn down the embedding size to print the
# embeddings and see what they look like. Printing and checking is a great way
# to see what's going on and confirm your intuitions.
#
# 2. There are many wrong implementations of word2vec out there. If you go
# looking for inspiration, be aware that some may be wrong.
#
# 3. Familiarizing youself with numpy will help.
#
# 4. The steps that are listed here are general guides and may correspond to one
# or more lines of code, depending on your programming style. Make sure you
# understand what the word2vec technique is supposed to be doing at that point
# to help guide your development.
#

import csv
import math
import pickle
import random
import sys
from collections import Counter
import numpy as np

# Hhelpful for computing cosine similarity
from scipy.spatial.distance import cosine

# This will make things go fast when we finally use it
from numba import jit

# Handy command-line argument parsing
import argparse

# Progress bar tracker
from tqdm import tqdm

# Sort of smart tokenization
from nltk.tokenize import RegexpTokenizer

# We'll use this to save our models
from gensim.models import KeyedVectors

#
# IMPORTANT NOTE: Always set your random seeds when dealing with stochastic
# algorithms as it lets your bugs be reproducible and (more importantly) it lets
# your results be reproducible by others.
#
random.seed(1234)
np.random.seed(1234)


class word2vec:
    def __init__(self, hidden_layer_size=50):

        self.hidden_layer_size = hidden_layer_size
        self.tokenizer = RegexpTokenizer(r'\w+')
        
        # These state variables become populated as the main() function calls
        #
        # 1. load_data()
        # 2. generate_negative_sampling_table()
        # 3. init_weights()
        #
        # See those functions for how the various values get filled in

        self.word_to_index = {} # word to unique-id
        self.index_to_word = [] # unique-id to word

        # How many times each word occurs in our data after filtering
        self.word_counts = Counter()

        # A utility data structure that lets us quickly sample "negative"
        # instances in a context. This table contains unique-ids
        self.negative_sampling_table = []
        
        # The dataset we'll use for training, as a sequence of unqiue word
        # ids. This is the sequence across all documents after tokens have been
        # randomly subsampled by the word2vec preprocessing step
        self.full_token_sequence_as_ids = []

        # These will contain the two weight matrices. W is the embeddings for
        # the center/target word and C are the embeddings for the context
        # words. You might see these called (W, V) or (W1, W2) in various
        # documentation too. These get initalized later in init_weights() once
        # we know the vocabulary size
        self.W = None
        self.C = None
        
    def tokenize(self, text):
        '''
        Tokenize the document and returns a list of the tokens
        '''
        return self.tokenizer.tokenize(text)

        
    def load_data(self, file_name=r"D:\Hu Jiaoyang Files\UM\21WINTER\630\HW\HW2\wiki-bios.med.txt", min_token_freq=5):
        '''
        Reads the data from the specified file as long long sequence of text
        (ignoring line breaks) and populates the data structures of this
        word2vec object.
        '''

        # Step 1: Read in the file and create a long sequence of tokens
        f = open(file_name, "r",encoding='UTF-8')
        token_list = []
        while (True):
            line = f.readline().lower()
            token_list += self.tokenize(line)
            if not line:
                break
        f.close()
        
        # Step 2: Count how many tokens we have of each type
        print('Counting token frequencies')
        c = Counter(token_list)

        # Step 3: Replace all tokens below the specified frequency with an <UNK>
        # token
        print("Performing minimum thresholding")
        key_list = []
        c['UNK'] = 0
        for i in tqdm(range(len(list(c.keys())))):
            k = list(c.keys())[i]
            if c[k] < min_token_freq:
                c['UNK'] += c[k]
                key_list.append(k)
        for i in tqdm(range(len(key_list))):
            k = key_list[i]
            c.pop(k, None)
        # Step 4: update self.word_counts to be the number of times each word
        # occurs (including <UNK>)
        print("load data: step 4")
        self.word_counts = c

        # Step 5: Create the mappings from word to unique integer ID and the
        # reverse mapping.
        #
        # HINT: the id-to-word mapping is easily represented as a list data
        # structure
        print("load data: step 5")
        self.word_to_index = dict( zip(list(c.keys()),list(range(len(c.keys())))) )
        self.index_to_word = list(c.keys()) # unique-id to word
        
        # Step 6: Compute the probability of keeping any particular token of a
        # word in the training sequence, which we'll use to subsample. This
        # avoids having the training data be filled with many overly common words
        print("load data: step 6")
        token_list_len = len(token_list)
        word_to_sample_prob = np.array([ (((np.sqrt(c[word]/token_list_len)/0.001)+1)*(0.001/(c[word]/token_list_len))) for word in list(c.keys())])
        # Step 7: process the list of tokens (after min-freq filtering) to fill
        # a new list self.full_token_sequence_as_ids where (1) we
        # probabilistically choose whether to keep each token based on the
        # subsampling probabilities and (2) all tokens are convered to their
        # unique ids for faster training.
        print("load data: step 7")
        for token in token_list:
            try:
                id = self.word_to_index[token]
            except:
                id = self.word_to_index['UNK']
            if (word_to_sample_prob[id]>random.uniform(0, 1)):
                    self.full_token_sequence_as_ids.append(id)
        # Transform the original input into a sequence of IDs while also
        # performing token-based subsampling based on the probabilities in
        # word_to_sample_prob. This effectively makes the context window larger
        # for some words by removing words that are common from a particular
        # context before the training occurs.
         
        print('Loaded all data from %s; saw %d tokens (%d unique)' \
              % (file_name, len(self.full_token_sequence_as_ids),
                 len(self.word_to_index)))
                
    def generate_negative_sampling_table(self, exp_power=0.75, table_size=1e6):
        '''
        Generates a big list data structure that we can quickly randomly index into
        in order to select a negative training example (i.e., a word that was
        *not* present in the context). 
        '''       
        
        # Step 1: Figure out how many instances of each word need to go into the
        # negative sampling table. 
        #
        # HINT: np.power and np.fill might be useful here
        weight_total = 0
        for v in self.word_counts.values():
            weight_total += np.power(v,exp_power)
        instance_table = (np.array([(np.power(v,exp_power)/weight_total) for v in self.word_counts.values()]) * table_size).astype(int)
        print("Generating sampling table")
        # Step 2: Create the table to the correct size. You'll want this to be a
        # numpy array of type int
        for i in range(len(instance_table)):
            new_table = np.zeros(instance_table[i])
            new_table.fill(i)
            self.negative_sampling_table = np.concatenate((self.negative_sampling_table, new_table), axis=None)
        # Step 3: Fill the table so that each word has a number of IDs
        # proportionate to its probability of being sampled.
        #
        # Example: if we have 3 words "a" "b" and "c" with probabilites 0.5,
        # 0.33, 0.16 and a table size of 6 then our table would look like this
        # (before converting the words to IDs):
        #
        # [ "a", "a", "a", "b", "b", "c" ]
        #


    def generate_negative_samples(self, cur_context_word_id, num_samples):
        '''
        Randomly samples the specified number of negative samples from the lookup
        table and returns this list of IDs as a numpy array. As a performance
        improvement, avoid sampling a negative example that has the same ID as
        the current positive context word.
        '''

        # Step 1: Create a list and sample from the negative_sampling_table to
        # grow the list to num_samples, avoiding adding a negative example that
        # has the same ID as teh current context_word
        results = []
        while(True):
            if (len(results)==num_samples):
                break
            index = np.random.randint(low = 0, high = len(self.negative_sampling_table))
            if (self.negative_sampling_table[index] == cur_context_word_id):
                continue
            else:
                results.append(self.negative_sampling_table[index])

        # Step 2: Convert the list of samples to numpy array and return it 
        return np.array(results).astype(int)

    def save(self, filename="gensim_KeyedVectors.txt"):
        '''
        Saves the model to the specified filename as a gensim KeyedVectors in the
        text format so you can load it separately.
        '''

        # Creates an empty KeyedVectors with our embedding size
        kv = KeyedVectors(vector_size=self.hidden_layer_size)        
        vectors = []
        words = []
        # Get the list of words/vectors in a consistent order
        for index, word in enumerate(self.index_to_word): 
            vectors.append(self.W[index].copy())
            words.append(word)
            
        # Fills the KV object with our data in the right order
        kv.add(words, vectors) 
        kv.save_word2vec_format(filename, binary=False)

    def init_weights(self, init_range=0.1):
        '''
        Initializes the weight matrices W (input->hidden) and C (hidden->output)
        by sampling uniformly within a small range around zero.
        '''

        # Step 1: Initialize two numpy arrays (matrices) for W and C by filling
        # their values with a random sample within the speified range.
        #
        # Hint: numpy.random has lots of ways to create matrices for this task
        vocab_size = len(self.index_to_word)
        self.W = np.random.uniform(low=-init_range, high=init_range, size=(vocab_size,self.hidden_layer_size) ) 
        self.C = np.random.uniform(low=-init_range, high=init_range, size=(self.hidden_layer_size,vocab_size) ) 

    def train(self, num_epochs=3, window_size=2, num_negative_samples=2,
              learning_rate=0.05, nll_update_iter=10000, max_steps=-2):
        '''
        Trains the word2vec model on the data loaded from load_data for the
        specified number of epochs.
        '''

        # Rather than compute the full negative log-likelihood (NLL), we'll keep
        # a running tally of the nll values for each step and periodically report them
        nll_results = []
        # This value keeps track of which step we're on. Since we don't update
        # when the center token is "<UNK>" we may skip over some ids in the
        # inner loop, so we need a separate step count to keep track of how many
        # updates we've done.
        step = 0
        #add synonyms
        f = open(r"D:\Hu Jiaoyang Files\UM\21WINTER\630\HW\HW2\synonyms.txt", "r",encoding='UTF-8')
        synonyms_list = []
        while (True):
            line = self.tokenize(f.readline().lower())
            line_ids = []
            for token in line:
                try:
                    line_ids.append(self.word_to_index[token])
                except:
                    pass
            if len(line_ids)> 1:
                synonyms_list.append(line_ids)
            if not line:
                break
        f.close()
        synonyms_set = set(map(frozenset, synonyms_list))
        # Iterate for the specified number of epochs
        for epoch in range(1, num_epochs+1):
            print("Beginning epoch %d of %d" % (epoch, num_epochs))           
            if epoch == 2:
                learning_rate = 0.005
            if epoch == 3:
                learning_rate = 0.0005
            # Step 1: Iterate over each ID in full_token_sequence_as_ids as a center
            # token (skipping those that are <UNK>) and predicting the context
            # word and negative samples
            #
            # Hint: this is a great loop to wrap with a tqdm() call so you can
            # see how long each epoch will take with a progress bar
            for i in tqdm(range(len(self.full_token_sequence_as_ids))):
                center_id = self.full_token_sequence_as_ids[i]
                try:
                    for sub_set in synonyms_set:
                        if center_id in sub_set:
                            center_id = random.sample(sub_set, 1)[0]
                except:
                    pass
                
                if (self.index_to_word[center_id] == 'UNK'):
                    continue

                # Periodically print the NLL so we can see how the model is converging
                if nll_update_iter > 0 and step % nll_update_iter == 0 and step > 0 and len(nll_results) > 0:
                    print("Negative log-likelihood (step: %d): %f " % (step, sum(nll_results)))
                    nll_results = []
                # Step 2: For each word in the window range (before and after)
                # perform an update where we (1) use the current parameters of
                # the model to predict it using the skip-gram task and (2)
                # sample negative instances and predict those. We'll use the
                # values of those predictions (i.e., the output of the sigmoid)
                # to update the W and C matrices using backpropagation.
                #
                # NOTE: this inner loop should call predict_and_backprop() which is
                # defined outside of the class. See note there for why.
                context = self.full_token_sequence_as_ids[(i-window_size):i]+self.full_token_sequence_as_ids[(i+1):(i+1+window_size)]
                for context_id in context:#should be something about context

                    # Step 3: Pick the context word ID


                    # Step 4: Sample negative instances 
                    negative_sample_ids = self.generate_negative_samples(cur_context_word_id = context_id, num_samples = num_negative_samples)

                    # Step 5: call predict_and_backprop. Don't forget to add the
                    # nll return value to nll_results to keep track of how the
                    # model is learning
                    nll = predict_and_backprop(self.W,self.C,learning_rate = learning_rate, center_id = center_id, context_id = context_id,negative_sample_ids = negative_sample_ids)
                    nll_results.append(nll)
                    
                step += 1
                if max_steps > 0 and step >= max_steps:
                    break
            if max_steps > 0 and step >= max_steps:
                print('Maximum number of steps reached: %d' % max_steps)
                break


    def get_neighbors(self, target_word):
        """ 
        Finds the top 10 most similar words to a target word
        """
        outputs = []
        for index, word in tqdm(enumerate(self.index_to_word), total=len(self.index_to_word)):
            similarity = self.compute_cosine_similarity(target_word, word)
            result = {"word": word, "score": similarity}
            outputs.append(result)
    
        # Sort by highest scores
        neighbors = sorted(outputs, key=lambda o: o['score'], reverse=True)
        return neighbors[1:11]

    def compute_cosine_similarity(self, word_one, word_two):
        '''
        Computes the cosine similarity between the two words
        '''
        try:
            word_one_index = self.word_to_index[word_one]
            word_two_index = self.word_to_index[word_two]
        except KeyError:
            return 0
    
        embedding_one = self.W[word_one_index]
        embedding_two = self.W[word_two_index]
        similarity = 1 - abs(float(cosine(embedding_one, embedding_two)))
        return similarity


#
# IMPORTANT NOTE:
# 
# These functions are specified *outside* of the word2vec class so that they can
# be compiled down into very effecient C code by the Numba library. Normally,
# we'd put them in the word2vec class itself but Numba doesn't know what to do
# with the self parameter, so we've pulled them out as separate functions which
# are easily compiled down.
#
# When you've gotten your implementation fully correct, uncomment the line above
# the function that reads:
#
#   @jit(nopython=True)
#
# Which will turn on the just-in-time (jit) Numba compiler for making this
# function very fast. From what we've seen, this makes our implemetnation around
# 300% faster which is a huge and easy speed up to run on the dataset.
#
# The gradient descent part requires the most "math" for your program and
# represents the hottest part of the code (since this is called multiple times
# for each context!). Speeding up this one piece can result in huge performance
# gains, so if you're feeling adventurous, try copying the code and then
# modifying it to see how you can make it faster. On a 2.7GHz i7 processor, tdqm
# reports about ~10k iterations/sec in our reference implementation.
#

@jit(nopython=True)
def predict_and_backprop(W, C, learning_rate, center_id, context_id,
                         negative_sample_ids):
    '''
    Using the center token (specified by center_id), makes a forward pass through
    the network to predict the context token (context_id) and negative samples,
    then backprops the error of those predictions to update the network and
    returns the negative log likelihood (Equation 1 in your homework) from the
    current preditions. W and C are the weight matrices of the network and IDs
    refer to particular rows of the matrices (i.e., the word embeddings of the
    target word and the context words!)

    '''

    #
    # GENERAL NOTE: There are many ways to implement this function, depending on
    # how fancy you want to get with numpy. The instructions/steps here are
    # intended as guides for the main tasks on what you have to do and may be
    # implemented as one line or more lines, depending on which methods you use
    # and how you want to write it. The important thing is that it works, not
    # how fast it is, so feel free to write it in a way that is understandable
    # to you. Often when you get to that point, you'll see ways to improve (but
    # first save a copy of your working code!).
    #
    
    nll = 0
    # Step 1: Look up the two vectors in W and C. Note that the row for the
    # center_id is effectively the hidden layer activation, h.

    #we use copy so that we dont make changes to the original weights just yet
    #h is the hidden layer activation
    h = np.copy(W[center_id])
    #v_i acts as an input to the second weight matrix C to produce the output neurons which will
    #be the same for all context words in the contexts window
    v_i =  np.copy(W[center_id]) 


    # Step 2: Look up the vectors for the negative sample IDs.
    #
    # NOTE: numpy supports multiple indexing (getting multiple rows at once) if
    # you want to use it
    v_T_neg_list = []
    for id in negative_sample_ids:
        v_T_neg_list.append(np.copy(C[:,id]))

    # Step 3: Compute the predictions for the context word and the negative
    # examples. We want the predictions of the context word to be near 1 and
    # those for the negative examples to be near 0.
    t_j = 1
    v_j_old_context = np.copy(C[:,context_id])
    v_j_new_context =  v_j_old_context - learning_rate *(sigmoid( np.dot(v_j_old_context,h) )-t_j) * h
    update_sum = (sigmoid( np.dot(v_j_old_context,h) )-t_j) * v_j_old_context

    t_j = 0
    v_j_new_neg_list = []
    for v_T_neg in v_T_neg_list:
        v_j_old_neg = np.copy(v_T_neg)
        v_j_new_neg = v_j_old_neg - learning_rate *(sigmoid( np.dot(v_j_old_neg,h) )-t_j) * h
        v_j_new_neg_list.append(v_j_new_neg)
        update_sum += (sigmoid( np.dot(v_j_old_neg,h) )-t_j)*v_j_old_neg

    #update for v_i

    v_i_new = v_i - learning_rate * (update_sum) 

    # Step 4: Compute the negative log likelihood
    #also distinguish between context and negative samples
    t_j = 1
    C[:,context_id] = v_j_new_context
    nll = (- np.log(sigmoid(np.dot(v_j_new_context,h))))
    
    t_j = 0
    for i in range(len(v_j_new_neg_list)):
        v_j_new_neg = v_j_new_neg_list[i]

        C[:,negative_sample_ids[i]] = v_j_new_neg
        nll +=  (-np.log( sigmoid((-1)*np.dot(v_j_new_neg,h)) ))
    # Step 5: Update the negative sample vectors to push their dot product with the
    # center word's vecter closer to zero.
    
    # Step 6: Now backprop all the way back to the center word's vector. Be sure to
    # update it based on the *old* values of the context vectors, not the
    # new values of the context vectors that you just updated!    
    W[center_id] = v_i_new
    return nll

    
@jit(nopython=True)
def sigmoid(x):
    '''
    Returns the sigmoid of the provided value
    '''
    return (1.0 / (1.0 + np.exp(-x)))

model = word2vec()

model.load_data()
model.generate_negative_sampling_table()
model.init_weights()
model.train()
model.save()
import pandas as pd
pairs_df = pd.read_csv("word_pairs_to_estimate_similarity.test.csv")
results_df = pd.DataFrame(columns=['pair_id','similarity'])
print("computing cosine similarity")
for i in range(len(pairs_df)):
    row = pairs_df.loc[i]
    results_df.loc[len(results_df)] = [i,model.compute_cosine_similarity(row['word1'],row['word2'])]
results_df["pair_id"] = results_df["pair_id"].astype(int)
results_df.to_csv("pairs.csv",index = False)

targets = ["january", "good", "the", "food", "engineering"]

for targ in targets:
    print("Target: ", targ)
    bestpreds = (model.get_neighbors(targ))
    for pred in bestpreds:
        print(pred["word"], ":", pred["score"])
    print("\n")

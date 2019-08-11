import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import csv
from torch.utils.data import Dataset, DataLoader
import nltk
#nltk.download('all')
from nltk.tokenize.treebank import TreebankWordDetokenizer


class ROCStories(Dataset):
    
    def __init__(self, root_dir, vocab=None, max_length=22, corpus='ROCStories'):
        
        """
        ROCStories data loader!!

        :param str root_dir: path to data directory
        :param vocab: A class to create a vocabulary and load query/response sentence pairs into memory
        :param max_length: maximum lenght of the list of tokens (We found it as (22) based on the lenght of the shortest sentence in our dataset).
        :param corpus: name of vocab object
        """    
        self.max_len = max_length
        
        # load in csv file and combine sentences
        file = pd.read_csv(root_dir, sep=",")
        file['AllTogether'] = file["sentence1"].str.cat(file[['sentence2', 'sentence3', 'sentence4', 'sentence5']], sep=' ')

        # convert the 'AllTogether' column to list
        self.sentences = file["AllTogether"].tolist()
        
        if vocab:
            self.voc = vocab
        else:
            self.voc = Voc(corpus)
            
            for x in self.sentences:
                # find tokens for each sentence using nltk
                tokens = nltk.word_tokenize(x)
                # add each word to the vocab list
                self.voc.addSentence(tokens[:self.max_len])

    def __len__(self):
        """Return the number of allowed sequences"""
        return len(self.sentences)
    
    def len_vocab(self):
        """Return the number of tokens in language"""
        return len(self.voc.index2word)

    def count_vocab(self, x):
        """Return the number of each token in language"""
        return self.voc.word2count(x)  
 
    def encode(self, text, trim=False):
        """
        Encode a string into token indexes

        :param str text: text to be encoded
        :return: list of token indexes
        """
    
        tokens = nltk.word_tokenize(text)
        
        # trim tokens to max length
        if trim:
            tokens = tokens[:self.max_len]
        
        try:
            token_list = []
            for i in tokens:
                token_list.append(self.voc.word2index[i])
                
        except KeyError:
            raise KeyError(f'The word "{i}" is not in the vocab list')
            
        return torch.LongTensor(token_list)  
    

    def decode(self, index):
        """
        Decode a series of indexes to a string

        :param tokens: list or tensor containing indexes
        :return: string of text 'de-encoded' back from tokens
        """
        #change to liost if it is a tensor
        if isinstance(index,torch.Tensor):
            index = index.tolist()
        # find the list of tokens for the list of given indexes
        list_tokens=[self.voc.index2word[x] for x in index]

        # reverse the tokenization with nltk and return the associated sentence
        return TreebankWordDetokenizer().detokenize(list_tokens)



    def __getitem__(self, idx):
        
        # get sentence from dataset, encode to token indexes
        # set max length for training purposes
        result = self.encode(self.sentences[idx], trim=True)
        
        
        return torch.LongTensor(result)
    
    
#--------------------------------------------------------------------------------------------------    
# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "<PAD>", SOS_token: "<SOS>", EOS_token: "<EOS>"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        #for word in sentence.split(' '):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1
            
    def __len__(self):
        return len(self.index2word)

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 # Count default tokens

        for word in keep_words:
            self.addWord(word)
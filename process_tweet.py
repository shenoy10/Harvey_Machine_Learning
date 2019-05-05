import numpy as np
import pandas as pd
from pprint import pprint

import gensim
from gensim.utils import simple_preprocess
from gensim import corpora, models

from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from symspellpy.symspellpy import SymSpell, Verbosity
from sklearn.model_selection import train_test_split


#create spell checker/word splitter
def create_symspell(max_edit_distance, prefix_length, freq_file_path):
    # create object
    sym_spell = SymSpell(max_edit_distance, prefix_length)
    
    # create dictionary using corpus.txt
    if not sym_spell.create_dictionary(freq_file_path):
        print("Corpus file not found")
        return None
    return sym_spell


def is_valid_token(w):
    special = ['<url>','<hashtag>', '<number>', '<user>']
    return w.isalpha() or w in special


def process_tweet(tweet, tknzr, sym_spell=None, advanced=False):
    st_1 = []
    for w in tknzr.tokenize(tweet):
        #remove retweet annotation if present:
        if w == 'RT':
            if advanced:
                st_1.append('rt')
        elif w[0] == '@':
            if advanced:
                st_1.append('<user>')
        #remove hashtag symbol
        elif w[0] == '#':
            st_1.append(w[1:])
        #replace link with LINK keyword
        elif w[:4] == 'http':
            st_1.append('<url>')
        elif w.isnumeric():
            if advanced:
                st_1.append('<number>')
        else:
            st_1.append(w)
    
    st_2 = []
    
    #remove stop words and punctuation, make everything lowercase
    if sym_spell != None:
        st_2 = [sym_spell.word_segmentation(w.lower()).corrected_string 
                for w in st_1 if w.isalpha() and not w.lower() in stop_words]
    elif advanced:
        st_2 = [w.lower() for w in st_1 if 
                not w.lower() in stop_words]
    else:
        st_2 = [w.lower() for w in st_1 if w.isalpha() and 
                not w.lower() in stop_words]
    
    #lemmatization (converts all words to root form for standardization)
    lem = WordNetLemmatizer()
    st_3 = list(map(lambda x: lem.lemmatize(x, pos='v'), st_2))
    
    #now do word segmentation/spell check
    return ' '.join(st_3)

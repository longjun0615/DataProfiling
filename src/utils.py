# -*- coding: utf-8 -*-
"""
Created on Wed May 19 14:59:07 2021

@author: SKIM78
"""


import pickle as pkl
import re
import string
import nltk
from nltk.stem import WordNetLemmatizer

import itertools
#import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'notebook')

nltk.set_proxy('internet.ford.com:83')
#from nltk import tokenize
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
np.set_printoptions(precision=3, linewidth=100)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


stemmer = nltk.PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()

analyzer = CountVectorizer().build_analyzer()

def stemming(doc):
    return (stemmer.stem(w) for w in analyzer(doc))


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;_]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
def text_prepare(text):

    text = text.lower()# lowercase text
    text_first = re.sub(REPLACE_BY_SPACE_RE,' ',text)# replace REPLACE_BY_SPACE_RE symbols by space in text
    text_second = re.sub(BAD_SYMBOLS_RE,'',text_first)#text.remove(BAD_SYMBOLS_RE)# delete symbols which are in BAD_SYMBOLS_RE from text
    text1 = ' '.join([w for w in text_second.split() if not w in STOPWORDS])# delete stopwords from text
    #text2 = stemmer.stem(text1)
    text2 = wordnet_lemmatizer.lemmatize(text1)
    return text2

#def test_preepare(text):
#
#    text = text.lower()# lowercase text
#    text_first = re.sub(REPLACE_BY_SPACE_RE,' ',text)# replace REPLACE_BY_SPACE_RE symbols by space in text
#    text_second = re.sub(BAD_SYMBOLS_RE,'',text_first)#text.remove(BAD_SYMBOLS_RE)# delete symbols which are in BAD_SYMBOLS_RE from text
#    text = ' '.join([w for w in text_second.split() if not w in STOPWORDS])# delete stopwords from text
#    return text
def mask_number(text):
    return re.sub(r'\d+', '_NUMBER', text)


def mask_timestamp(text):
    re_time1 = '\d{1,2}[:.]\d{2}(?:am|pm|AM|PM)'
    re_time2 = '\d{1,2}[:.]\d{2}'
    re_time3 = '\d{1,2}(?:am|pm|AM|PM)'
    rec_time = re.compile(re_time1 + '|' + re_time2 + '|' + re_time3)
    return re.sub(rec_time, '_TIME', text)


def mask_all(text):
    text = mask_number(text)
    text = mask_timestamp(text)
    text = text_prepare(text)
    return text


def train_and_test(steps, X_train, X_test, y_train, y_test):
    pipeline = Pipeline(steps)
    folds = 10
    xval_score = cross_val_score(pipeline, X_train, y_train, cv=folds, n_jobs=-1)
    
    xv_min = np.min(xval_score)
    xv_max = np.max(xval_score)
    xv_mean = np.mean(xval_score)
    xv_std = np.std(xval_score)
    print('{} fold Cross Validation Score: <{:.2f}, {:.2f}>; µ={:.2f}'.format(folds, xv_min, xv_max, xv_mean))
    pipeline = pipeline.fit(X_train, y_train)
    print('Score on test set: {:.2f}'.format(pipeline.score(X_test, y_test)))
    return pipeline


def tag_message(pipeline, message):
    y_pred = pipeline.predict([message])[0]
    print('{:>20} | {}'.format(dict_classes[y_pred], message))
    

def multitag_message(pipeline, message):
    y_pred = pipeline.predict([message])[0]
    tags = [dict_classes[i+1] for i, yi in enumerate(y_pred) if yi == 1]
    # Remove `undefined` tag if the list contains other tags as well.
    if len(tags) > 1 and dict_classes[60] in tags:
        del tags[tags.index(dict_classes[60])]
    print('{:>30} | {}'.format('['+'] ['.join(tags)+']', message))



def sentence_tokenizer(text, verbose=False):
    # Some input checking.
    if not isinstance(text, str):
        print('[!] Input type should be a string, not a {}'.format(type(text)))
        exit(1)
        
    # Split sentences with NLTK
    text_list_1 = nltk.sent_tokenize(text)
    
    # Split sentences with our POS tagging method
    text_list_2 = []
    for text in text_list_1:
        text_list_2 += c_sentence_tokenizer(text, verbose)
    
    return text_list_2


def c_sentence_tokenizer(text, verbose=False):
    # container for final result
    new_split = []
    
    # Split sentences by a comma, 'and' and 'or'.
    text_list = re.split(',| and | or | but | \(', text)
    
    # Remove white spaces and empty string elements from the list
    text_list = [x.strip() for x in text_list]
    text_list = list(filter(None, text_list))
        
    # Append first list element to the new list.
    new_split.append(text_list[0])
    
    # Check if the splits are valid sentences. If not, glue the parts together again.
    for index in range(1, len(text_list)):
        
        # Keep the split if both parts of the sentences contain a verb.
        if find_verb(text_list[index-1], verbose) and find_verb(text_list[index], verbose):
            new_split.append(text_list[index])
        # Glue the parts together again, since POS requirements are not met.
        else:
            new_split[-1] += ' ' + text_list[index]
    
    if verbose:
        print('[.] Input sentence:')
        print('    ', text)
        print('[.] Output sentence(s):')
        print('    ', new_split)
    return new_split
    

def find_verb(sentence, verbose=False):
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    if verbose:
        print(pos_tagged)
    if 'VB' in [tag[1][:2] for tag in pos_tagged]:
        return True
    return False
#
#def plot_confusion_matrix(cm, classes,
#                          normalize=False,
#                          title='Confusion matrix',
#                          cmap=plt.cm.Blues):
#   
#    if normalize:
#        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#        print("Normalized confusion matrix")
#    else:
#        print('Confusion matrix, without normalization')
#
#    plt.imshow(cm, interpolation='nearest', cmap=cmap)
#    plt.title(title)
#    plt.colorbar()
#    tick_marks = np.arange(len(classes))
#    plt.xticks(tick_marks, classes, rotation=45)
#    plt.yticks(tick_marks, classes)
#
#    fmt = '.2f' if normalize else 'd'
#    thresh = cm.max() / 2.
#    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#        plt.text(j, i, format(cm[i, j], fmt),
#                 horizontalalignment="center",
#                 color="white" if cm[i, j] > thresh else "black")
#
#    plt.tight_layout()
#    plt.ylabel('True label')
#    plt.xlabel('Predicted label')

def filter_threshold(x):
    if np.any(x>.09):
        result=np.argmax(x)
    else:
        result=0    
    return result

def get_model_version(model_name: str) -> str:
    """
    model_name : "name_v#.#.#" -> version = #.#.#
    """
    model_name_split = model_name.split('_v')
    if len(model_name_split) == 1:
        return '0'
    else:
        return model_name_split[1]

def get_latest_model(savepath=None):
    if savepath is None:
        if "src" in os.getcwd().lower():
            path = "tsad_data/"
        else:
            path = "src/tsad_data/"
    else:
        path = savepath

    model_names = os.listdir(path)
    version_list = [int(get_model_version(model_name)) for model_name in model_names]
    arg_latest = np.argmax(version_list)

    return model_names[arg_latest]


#%%
""" custom vectorizer """
    
class StemmingVectorizer(CountVectorizer): 
    
    def build_analyzer(self):
        
        stop_words = self.get_stop_words()
        
        def analyser(doc):
            
            preprocess = self.build_preprocessor() # load preprocessing
            tokenize = self.build_tokenizer()
            doc_clean = preprocess(doc) # preprocess
            
#            stemming_tokens = [stemmer.stem(w) for w in analyzer(doc_clean)] # stemming w/ default CountVectorizer analyzer
            stemming_tokens = [stemmer.stem(w) for w in tokenize(doc_clean)] # stemming w/ tokenizer
#            try:
#                stemming_tokens = [str(w2n.word_to_num(w.text)) if w.pos_ == 'NUM' else stemmer.stem(w.text) for w in nlp(doc_clean)] # stemming & word2num w/ spacy
#            except:
#                stemming_tokens = []
#                print(doc_clean)

            return(self._word_ngrams(stemming_tokens, stop_words)) # apply stop_words & ngram_range
            
        return(analyser)
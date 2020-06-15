# -*- coding: UTF-8 -*-

# Clean NLP

import pandas as pd
import numpy as np
import nltk
from textblob import TextBlob

def preprocess_text_dataframe(df, column):
    '''
    Clean NLP strings from a dataframe
    '''
    df_processed = df
    stopWords = set(stopwords.words('english'))
    lemmatizer = nltk.stem.WordNetLemmatizer()
    df_processed[column] = df[column].str.replace('[^\w\s]','')
    df_processed[column] = df[column].str.lower()
    df_processed[column] = df[column].str.replace('\d+', '')
    df_processed[column] = df_processed[column].apply(nltk.word_tokenize)
    df_processed[column] = df_processed[column].apply(lambda x: [item for item in x if item not in stopWords])
    df_processed[column] = df_processed[column].apply(lambda x: [lemmatizer.lemmatize(w) for w in x])
    return df_processed

def polarity(text):
    '''
    Create the polarity from a textblob
    '''
    blob = TextBlob(text)
    polarity = blob.sentiment[0]
    return polarity

def subjectivity(text):
    '''
    Create the subjectivity from a textblob
    '''
    blob = TextBlob(text)
    subjectivity = blob.sentiment[1]
    return subjectivity

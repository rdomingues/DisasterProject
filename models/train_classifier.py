import sys
import joblib
import nltk
from sklearn.utils import parallel_backend

nltk.download(['punkt', 'wordnet','stopwords'])

import pandas as pd
import sqlite3
from sqlalchemy import create_engine


import re
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql("SELECT * from MessageTable", engine)
    X = df.message
    y = df.iloc[:, 4:]

    return X,y, y.columns


def tokenize(text):

    word_tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    stop_words = set(stopwords.words('english'))

    filtered_sentence = [w for w in word_tokens if not w in stop_words and w.isalpha()]

    clean_tokens = []
    for tok in filtered_sentence:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    stop = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()

    stop_words = []
    for tok in stop:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        stop_words.append(clean_tok)

    parameters = {
        'tfidf__norm': ['l2', 'l1'],
        'vect__stop_words': [stop_words, None],
        'clf__estimator__learning_rate': [0.1, 0.5, 1, 2],
        'clf__estimator__n_estimators': [50, 60, 70],
    }

    cv = GridSearchCV(pipeline, parameters, verbose=2, n_jobs=1)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    #predict
    y_pred = model.predict(X_test)
    # print the metrics
    for i, col in enumerate(category_names):
        print('{} category metrics: '.format(col))
        print(classification_report(Y_test.iloc[:, i], y_pred[:, i]))


def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        #with parallel_backend('threading'):
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
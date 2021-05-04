import pandas as pd
import sqlite3

from sklearn.utils import parallel_backend
from sqlalchemy import create_engine

import sys
import joblib

import re
import nltk
from sklearn.base import BaseEstimator, TransformerMixin

nltk.download(['punkt', 'wordnet','stopwords'])
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords



from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import train_test_split


def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql("SELECT * from MessageTable", engine)
    X = df.message
    y = df.iloc[:, 4:]

    return X,y, y.columns


def tokenize(text):
    """
      Tokenize method

       This method tokenize the text in argument
       First remove urls with a placeholder, then remove non words and stop words
       Then tokenize the setence and lemmatize them
       """
    # Replace all urls with a urlplaceholder string
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # Extract all the urls from the provided text
    detected_urls = re.findall(url_regex, text)

    url_place_holder_string ="urlPlaceHolder"

    # Replace url with a url placeholder string
    for detected_url in detected_urls:
        text = text.replace(detected_url, url_place_holder_string)

    word_tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    stop_words = set(stopwords.words('english'))

    filtered_sentence = [w for w in word_tokens if not w in stop_words and w.isalpha()]

    clean_tokens = []
    for tok in filtered_sentence:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class

    This class extract the starting verb of a sentence,
    creating a new feature for the ML classifier
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            #print(pos_tags)
            if len(pos_tags)>0:
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return True
        return False

    # Given it is a tranformer we can return the self
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_model():
    """
     Create model with pipeline and parameters
     return GridSearchCV model
    """

    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf',  MultiOutputClassifier(AdaBoostClassifier()))
    ])

    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        #'features__text_pipeline__tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 200],
        'clf__estimator__learning_rate': [0.1, 1, 2],
        'features__transformer_weights': (
            {'text_pipeline': 1, 'starting_verb': 0.5},
           # {'text_pipeline': 0.5, 'starting_verb': 1},
            {'text_pipeline': 0.8, 'starting_verb': 1},
        )
    }

    cv = GridSearchCV(pipeline, parameters, verbose=2, n_jobs=-1)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
        evaluate the model using classification_report
        the method receive:
        model: the model to evalute
        X_test: the subset of messages
        Y_test: the subset of classification
        category_names: The categories name
    """


    y_pred = model.predict(X_test)
    # print the metrics
    for i, col in enumerate(category_names):
        print('{} category metrics: '.format(col))
        print(classification_report(Y_test.iloc[:, i], y_pred[:, i]))


def save_model(model, model_filepath):
    """
        save the model to disk
        the methor receive:
        model: the model to save
        model_filepath: path to save the file
    """
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
        with parallel_backend('multiprocessing'):
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
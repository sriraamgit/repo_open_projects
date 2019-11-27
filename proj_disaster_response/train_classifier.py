#Load the required packages
import sys
import re
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
import sqlite3
from sqlalchemy import *
from sqlalchemy import create_engine
from sqlalchemy import Table
from sqlalchemy import MetaData
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.multiclass import OneVsRestClassifier
import pickle

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def load_data(database_filepath):
    '''
    function loads data from database and seperates into two dataframes
    Input: database filepath
    Output: df dataframe, X dataframe, Y dataframe, column names of Y as category_names
    '''
    #Loading data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df=pd.read_sql_table('disaster_response', engine)
    #seperating into two dataframes
    X = df['message']
    Y = df[['related', 'request', 'offer', 'aid_related', 'medical_help',
            'medical_products', 'search_and_rescue', 'security', 'military',
            'child_alone', 'water', 'food', 'shelter', 'clothing', 'money',
            'missing_people', 'refugees', 'death', 'other_aid',
            'infrastructure_related', 'transport', 'buildings', 'electricity',
            'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
            'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
            'other_weather', 'direct_report']]
    # extracting Y dataframe column names 
    category_names= Y.columns
    
    return df, X, Y, category_names

def tokenize(text):
    '''
    function creates tokens from text
    Input: text of messages column
    Output: tokenized clean_tokens
    '''
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    detected_urls = re.findall(url_regex, text)
    #Replace url's with urlplaceholder
    for url in detected_urls:
        text=text.replace(url, "urlplaceholder")
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens
    pass
    
    
def build_model():
    '''
    function for building model and pipeline using GridSearch and parameters
    Input: none
    Ouput: cv, used as model later
    '''
    #As you can see in workspace results f1-score was good when AdaBoost classifier used
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(AdaBoostClassifier(random_state=0)))
    ])
   
    parameters = {
        #"vect_ngram_range":((1, 1), (1, 2)),
        #"vect_max_df":(0.5, 0.75),
        #"vect_max_features":(None, 5000, 10000),
        "tfidf__use_idf":(True, False),
        "clf__estimator__n_estimators": [10, 20]
        #"clf_min_samples_split": [2, 3, 4]
    }   
    cv = GridSearchCV(pipeline, param_grid = parameters, n_jobs=4, verbose=2)
    
    return cv
    
    
def evaluate_model(model, X_test, Y_test, category_names, Y):
    '''
    function evaluates trained model and predicts test model
    Input: model built earlier, X_test messages in test split, Y_test categories in test split,
           category_names for labels, Y for same
    Output: Overall Classification report, Classification report for each column(metrics like
            precision, recall, f_score)    
    '''
    #classification report for the entire dataset
    Y_pred = model.predict(X_test)
    print(classification_report(np.ravel(Y_test), np.ravel(Y_pred)))
           
    #classification report got by looping through each column after applying grid search
    Y_preds=pd.DataFrame(Y_pred)
    
    for i, col in enumerate(Y.columns):
        print("Colummn : {}".format(col))
        Y_true = Y_test.values[:, i]
        Y_predss = Y_preds.values[:, i]
        target_names = ['is_{}'.format(col), 'not_{}'.format(col)]
        print(classification_report(Y_true, Y_predss, target_names=target_names))
    pass          
    

def save_model(model, model_filepath):
    '''
    function saves the model as pickle and another using joblib, these files used in app folder
    Input: model, model_filepath 
    Output: none
    '''    
    pickle.dump(model, open(model_filepath, 'wb'))
    print('pickle model saved in :', model_filepath)
    joblib.dump(model, 'classifier.pkl')
    print('joblib model saved in :', 'classifier.pkl')
    pass      
    

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        df, X, Y, category_names = load_data(database_filepath)
                
        for message in X[:5]:
            tokens = tokenize(message)
            print(message)
            print(tokens, '\n')
       
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)
        print('Trained model saved!')        
             
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names, Y)
        print('Model Evaluated')
       
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)
        
        print('Trained model saved after evaluating!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
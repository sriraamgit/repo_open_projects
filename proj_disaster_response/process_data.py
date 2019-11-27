#importing all the required packages
import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine
from sqlalchemy import MetaData
from sqlalchemy import Table
from sqlalchemy import *

def load_data(messages_filepath, categories_filepath):
    '''
    function reads two data files and returns files seperately and merged dataframe
    Input: messages.csv, category.csv files
    Output: messages and categories dataframe and merged dataframe df
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df=messages.merge(categories, how='outer', on='id')
    return messages, categories, df
    pass 

def clean_data(df):
    '''
    function to clean the dataframe
    Input: df dataframe
    output: cleaned df dataframe 
    '''
    #Split categories column
    categories = df['categories'].str.split(';', n=-1, expand=True)
    #Extract first row names
    row_nam=categories.loc[0,:]
    #Change the row names extracted to string datatype
    row_nam=row_nam.astype(str)
    #Extract all characters leaving the last two characters
    row_nam=row_nam.str.slice(0,-2,1)
    #Name the columns with the above extracted names
    categories.columns=row_nam
    category_colnames = row_nam      
    # change the datatypes of colummn vaues also to string and
    #set each value in the dataframe to be the last character of the string
    categories.astype(str)
    for column in categories:
        categories[column] = categories[column].str.slice(-1)
    # convert columns from string to numeric
    for column in categories:
        categories[column] = pd.to_numeric(categories[column])
        
    new_categories = categories    
    # drop the original categories column from `df` dataframe and rename the dataframe
    df.drop(['categories'], axis=1, inplace=True)
    df_new = df
    # rename the categories dataframe which we split into 36 columns and 
    #merge with the new dataframe in which we dropped the oriinal categories column
    new_categories = categories
    new_categories['id'] = df_new['id']
    df = df_new.merge(new_categories, how='outer', on='id')
    #Drop duplicates    
    df.duplicated(keep='first')
    df.drop_duplicates()
    df.drop_duplicates(subset='message', keep=False, inplace=True)
    df=df[df['related'] != 2]
    
    return df
    
   
def save_data(df, database_filepath):
    '''
    function saves the cleaned dataframe to a sqlite database named DisasterResponse
    Input: cleaned dataframe df and database filepath
    Output: none
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df.to_sql('disaster_response', engine, if_exists='replace', index=False)
    pass


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        messages, categories, df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        n=df.related.value_counts()
        print(n)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        
        print('Cleaned data saved to database!')
                
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
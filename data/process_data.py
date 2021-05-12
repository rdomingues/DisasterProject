import sys
import pandas as pd
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    load_data method: load data from two sources to a dataframe
    :messages_filepath messages_filepath: filepath of messages dataset
    :categories_filepath categories_filepath: filepath of categories dataset
    :return: dataframe merged
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath, sep=",")

    # load categories dataset
    categories = pd.read_csv(categories_filepath, sep=",")

    df = pd.merge(messages, categories, on="id")

    return df

def clean_data(df):
    '''
    Clean data, removing NA values, creating a boolean column for each category and append to dataframe. Remove duplicates
    :param df: dataframe with data to clean
    :return: dataframe after cleaned
    '''
    df = df.dropna(axis=0, subset=['message'])

    categories = df["categories"].str.split(";", expand=True)

    row = categories.iloc[0:1]

    getDescription = lambda col: col.str[:-2]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    # category_colnames = row.apply(getDescription)

    category_colnames = (row.apply(getDescription))
    category_colnames = list(category_colnames.iloc[0])
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        # convert column from string to numeric
        categories[column] = categories[column].str[-1].astype(int)
        categories = categories.dropna(axis=0, subset=[column])

        #categories = categories.drop(categories[categories[column] > 1].index)
        #categories = categories[categories[column] < 2]

    #categories = categories[(categories < 2).all(axis=1)]

    df = df.drop(columns=["categories"])

    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df = df.drop_duplicates(subset="id")
    df = df.dropna(axis=0, subset=['message'])

    for column in categories:
        df = df[df[column] < 2]

    return df

def save_data(df, database_filename):
    '''
    save the data to a sqlite database file
    :param df: dataframe to save in database
    :param database_filename: the database name to save
    '''
    engine = create_engine('sqlite:///'+database_filename+'.db')
    df.to_sql('MessageTable', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
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
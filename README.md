# Disaster Response Pipeline Project

### Table of Contents
1. [Instructions](#Instructions)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Licensing, Authors, and Acknowledgements](#licensing)

## Instructions <a name="Instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


## Project Motivation<a name="motivation"></a>

The scope of this analysis is to complete Udacity Data Science Nanodegree. I’m proposing to create a disaster analysis to detect what is the subject of the help request.

## File Descriptions <a name="files"></a>

There are several folders in the project:

#### JupyterNotebooks
Two Jupyter notebooks to analyse the dataset and create a first model

#### data
Where is located the python file to read the csv dataset

#### models
Where is located the model with a learning model to understand and classify the messages

#### app
Flask app to try the message tool and see some graphs

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

MIT License

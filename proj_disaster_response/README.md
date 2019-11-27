#project_disaster_response

Summary of the Project: The overall summary in one line of this project is that, we analyze disaster data to build a model for an 
API that classifies disaster messages and also display some visualizations as the picture gives a better idea. The dataset consists of 
real messages that were sent during disaster events.I created a machine learning pipeline and wrote clean organised code to categorize 
these events so that the messages can be sent to any disaster relief agency. An emergency worker can input a new message and get 
classification results in the 36 categories described in this project.

How to run the python scripts and web app: a. First we run ETL code as python data/process_data.py data/disaster_messages.csv 
data/disaster_categories.csv data/DisasterResponse.db b. Then we run M pipeline fr building, training and saving classifier python
models/train_classifier.py data/DisasterResponse.db models/classifier.pkl c. Finally to have a look at the visualizations and classify
messages navigate to app folder and type python run.py

Explanation of files in the Repository: 
I have created two script files process_data.py and train_classifier.py and added some code 
in run.py file to get the visualizations and to classify messages into 36 categories. The first script file
process_data.py has three functions, the first function load_data() loads data files and merges them into a single dataframe. 
The second function clean_data() cleans and shapes the dataframe in the way required. The third function save_data() saves the data the
sqlite database with the name Disaster_Response.db The second script file train_classifier.py has the function load_data() to load
the data from database, the second function tokenize() creates tokens from text in messages dataframe, the third function
build_model() is for building model and pipeline using grid search and parameters, the fourth function evaluate_model()
evaluates a trained model and predicts test model and another function save_model() saves the model as a pickle in the file
named classifier.pkl and this file is later used in run.py file in app folder. By running run.py file we can see the visualizatins 
created by me. By typing the message in the textbox and clicking on "classify message", we can see the 36 categories described in 
this project.

Project Definition: Resilient Distributed Datasets is a fundamental Datastructure of Spark. It is an immutable distributed 
collection of objects. Each dataset in RDD is divided into logical partitions which may be computed on different nodes of the cluster.
In this project we manipulate large and realistic datasets with Spark to engineer relevant features for predicting churn. We use Spark
MLlib to build machine learning models with large datasets, far beyond what could be done with non-distributed technologies like
scikit-learn.

Motivation: Keeping in view the large data being generated these days, I am interested to learn how to load large datasets into Spark 
and manipulate them using Spark SQL and Spark Dataframes, use the machine learning APIs within Spark ML to build and tune models. In 
this project I integrated what I have learnt till now with Spark.This new skills I am learning gave me the driving force to complete 
the project.

Analysis: The project is divided into sections like: Load and Clean Dataset: I have loaded and cleaned the dataset, checking for
invalid or missing data - for example, records without userids or sessionids. Exploratory Data Analysis: I have performed EDA by 
loading a small subset of the data and doing basic manipulations within Spark. Created a column Churn. Performed some exploratory 
data analysis to observe the behavior for users who stayed vs users who churned. By exploring aggregates on these two groups of users, 
observing how much of a specific action they experienced per a certain time unit or number of songs played. Feature Engineering: I have
written a script to extract the necessary features from the smaller subset of data and ensured that the script is scalable.
Modeling: Performed Split of the full dataset into train, test sets. For testing out several of the machine learning methods learned
previously I used NaiveBayes and RandomForestClassifier. From my previous projects I have done in the classroom, RFC was giving good
F1-Score when hyperparameter tuning was performed correctly, So finally I have taken RFC to optimize the F1-Score metric and plotted
the feature importances.

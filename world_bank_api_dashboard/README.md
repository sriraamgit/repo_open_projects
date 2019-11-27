Summary of the project:
    Some companies provide public facing API's so that we can pull the data.
In this project we used that and pulled data directly from the API and then visualized using Plotly.
This flask app visualizes data from the world bank API.

How to run the app:
    To display the visualizations on the web page run the file worldbank.py

Explanation of some of the files in the Repository:
    index.html: Some visualizations are displayed using this file. A filter is also provided 
    to filter the selected countries.
    routes.py: This file is used to parse the POST, GET requests and provides list of countries for filtering.
    worldbank.py: In this file there is code to run the application on the local development server.

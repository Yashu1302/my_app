import streamlit as st
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
# Add a title to the web app
st.title("Sentiment Analysis")

# Load the data (similar to creating a dataframe)
#var = pd.read_table("C:\Users\YASHWANTH ADMIN\OneDrive\Documents\coding\Restaurant_Reviews (1).tsv")
# Load the data (using forward slashes in the path)
var = pd.read_table("C:/Users/YASHWANTH ADMIN/OneDrive/Documents/coding/Restaurant_Reviews (1).tsv")
# Divide data into input and output
x = var.Review     # input
y = var.Liked      # output

from sklearn.pipeline import make_pipeline
model = make_pipeline(CountVectorizer(), SVC())
model.fit(x, y)

# Input review
x_review = st.text_input('Enter Review')
if x_review:
    # Predict the output
    y_pred = model.predict([x_review])

    # Print the predicted output
    op = ['IT IS A NEGATIVE REVIEW 😢', 'IT IS A POSITIVE REVIEW 😊']
    st.title(op[y_pred[0]])
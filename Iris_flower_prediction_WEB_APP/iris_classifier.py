import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Simple Iris flower detection web app.
The app preict the flower type
""")

st.sidebar.header('user input parameter')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 1.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length' : sepal_length,
            'sepal_width' : sepal_width,
            'petal_length' : petal_length,
            'petal_width' : petal_width}
    features = pd.DataFrame(data, index=['input'])
    return features

df = user_input_features()

st.subheader('User input parameter Table')
st.write(df)

iris = datasets.load_iris()
X = iris.data
y = iris.target

clf = RandomForestClassifier()
clf.fit(X, y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class label and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])

st.write('prediction probablity')
st.write(prediction_proba)
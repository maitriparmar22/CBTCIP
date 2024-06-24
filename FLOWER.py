# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 11:48:25 2024

@author: Shreyash Parmar
"""

import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("Iris Flower.csv.csv")

# Drop the 'Id' column if it exists
if 'Id' in df.columns:
    df.drop(columns=['Id'], inplace=True)

# Convert the 'Species' column to a numerical representation
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])

# Define features and labels
X = df.drop(columns=['Species'])
Y = df['Species']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

# Train the models
logistic_model = LogisticRegression(solver='lbfgs', max_iter=1000)
logistic_model.fit(X_train, Y_train)

knn_model = KNeighborsClassifier()
knn_model.fit(X_train, Y_train)

decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X_train, Y_train)

# Streamlit app
st.title('Iris Flower Prediction')

st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('SepalLengthCm', float(df.SepalLengthCm.min()), float(df.SepalLengthCm.max()))
    sepal_width = st.sidebar.slider('SepalWidthCm', float(df.SepalWidthCm.min()), float(df.SepalWidthCm.max()))
    petal_length = st.sidebar.slider('PetalLengthCm', float(df.PetalLengthCm.min()), float(df.PetalLengthCm.max()))
    petal_width = st.sidebar.slider('PetalWidthCm', float(df.PetalWidthCm.min()), float(df.PetalWidthCm.max()))
    data = {'SepalLengthCm': sepal_length,
            'SepalWidthCm': sepal_width,
            'PetalLengthCm': petal_length,
            'PetalWidthCm': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('User Input parameters')
st.write(input_df)

# Ensure the order of columns in input_df matches the training data
input_features = input_df[X_train.columns]

# Display the predictions
st.subheader('Prediction')

logistic_pred = logistic_model.predict(input_features)
knn_pred = knn_model.predict(input_features)
decision_tree_pred = decision_tree_model.predict(input_features)

st.write(f"Logistic Regression Prediction: {le.inverse_transform(logistic_pred)[0]}")
st.write(f"KNN Prediction: {le.inverse_transform(knn_pred)[0]}")
st.write(f"Decision Tree Prediction: {le.inverse_transform(decision_tree_pred)[0]}")

# Display model accuracy
st.subheader('Model Accuracy')

logistic_accuracy = logistic_model.score(X_test, Y_test) * 100
knn_accuracy = knn_model.score(X_test, Y_test) * 100
decision_tree_accuracy = decision_tree_model.score(X_test, Y_test) * 100

st.write(f'Logistic Regression Accuracy: {logistic_accuracy:.2f}%')
st.write(f'KNN Accuracy: {knn_accuracy:.2f}%')
st.write(f'Decision Tree Accuracy: {decision_tree_accuracy:.2f}%')


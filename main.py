

import streamlit as st

from sklearn import datasets

from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image


header = st.beta_container()
dataset = st.beta_container()
visualize = st.beta_container()
model_training = st.beta_container()


with header:
    st.title('Welcome to Breast Cancer classification App')
    st.text('In this App you can classifiy breast cancer with several algorithms')

    image = Image.open('breast_cancer.jpeg')
    st.image(image, use_column_width=True)
with dataset:
    st.header('Breast Cancer dataset ')
    st.text('This dataset is part of sklearn datasets')

    data = load_breast_cancer()
    st.write(data.keys())
    # describe the dataset
    if st.checkbox('Describe the dataset'):
        st.write(data.DESCR)

    main_df = pd.DataFrame(data.data, columns=data.feature_names)
    main_df['target'] = data.target
    st.write(main_df['target'].value_counts())
    st.write(main_df.head())


with visualize:

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.header('Visualize Dataset')
    st.write('Correlations between features')
    features_mean = list(main_df.columns[1:11])
    plt.figure(figsize=(10, 10))
    sns.heatmap(main_df[features_mean].corr(), annot=True,
                square=True, cmap='coolwarm')
    st.pyplot()

    # radius = main_df['mean radius', 'radius_se', 'radius_worst', 'target']
    # sns.pairplot(radius, hue='target', palette="husl",
    #              markers = ["o", "s"], size = 4)

    # st.pyplot()

with model_training:
    st.header('Here you can train your model')
    st.text('You can choose several algorithms')

    # store features
    X = data.data
    y = data.target

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    classifier = st.selectbox(
        'Choose your classifier algorithm', ('Random Forest', 'KNN', 'SVM'))

    if classifier == 'KNN':

        k = st.slider('Choose number of neighbours', min_value=1, max_value=20)

        KNN = KNeighborsClassifier(n_neighbors=k)
        KNN.fit(X_train, y_train)
        # score_KNN = KNN.score(X_test, y_test)
        # st.write('Accuracy score of KNN model is {}'.format(score_KNN))

        st.write(classification_report(y_test, KNN.predict(X_test)))
        st.write(accuracy_score(y_test, KNN.predict(X_test)))

    if classifier == 'SVM':
        c = st.slider(label='Chose value of C',
                      min_value=0.1, max_value=10.0)

        SVM = SVC(C=c)
        SVM.fit(X_train, y_train)
        # score_SVM = SVM.score(X_test, y_test)
        # st.write('Accuracy score for SVM is {}'.format(score_SVM))

        st.write(classification_report(y_test, SVM.predict(X_test)))
        st.write(accuracy_score(y_test, SVM.predict(X_test)))

    if classifier == 'Random Forest':
        max_depth = st.slider('max_depth', 2, 10)
        n_estimators = st.slider('n_estimators', 1, 100)

        forest = RandomForestClassifier(
            max_depth=max_depth, n_estimators=n_estimators, random_state=1)

        forest.fit(X_train, y_train)
        # forest_score = forest.score(X_test, y_test)
        # st.write('Accuracy score for Random Forest is {}'.format(forest_score))

        st.write(classification_report(y_test, forest.predict(X_test)))
        st.write(accuracy_score(y_test, forest.predict(X_test)))

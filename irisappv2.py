import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

iris_df = pd.read_csv("iris-species.csv")

iris_df["Label"] = iris_df["Species"].map({"Iris-setosa":0,"Iris-virginica":1,"Iris-versicolor":2})

x = iris_df[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]
y = iris_df["Label"]

X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.33,random_state=42)

model = SVC(kernel="linear")
model.fit(X_train,Y_train)
score = model.score(X_train,Y_train)

modelRF = RandomForestClassifier(n_jobs=-1,n_estimators=100)
modelRF.fit(X_train,Y_train)
scoreRF = modelRF.score(X_train,Y_train)

modelLR = LogisticRegression(n_jobs=-1)
modelLR.fit(X_train,Y_train)
scoreLR = modelLR.score(X_train,Y_train)

@st.cache()
def prediction(model,SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm):
	species = model.predict([[SepalWidthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm]])
	species = species[0]
	if species==0:
		return "Iris-setosa"
	elif species == 1:
		return "Iris-virginica"
	else:
		return "Iris-versicolor"


st.sidebar.title("IRIS PREDICTOR")
SepalLengthCm = st.sidebar.slider("Sepal Length",float(iris_df["SepalLengthCm"].min()),float(iris_df["SepalLengthCm"].max()))
PetalLengthCm = st.sidebar.slider("Petal Length",float(iris_df["PetalLengthCm"].min()),float(iris_df["PetalLengthCm"].max()))
SepalWidthCm = st.sidebar.slider("Sepal Width",float(iris_df["SepalWidthCm"].min()),float(iris_df["SepalWidthCm"].max()))
PetalWidthCm = st.sidebar.slider("Petal Width",float(iris_df["PetalWidthCm"].min()),float(iris_df["PetalWidthCm"].max()))

select_box = st.sidebar.selectbox("Classifier",("SVM","Random Forest","LogisticRegression"))
button1 = st.sidebar.button("Predict Button")
if button1:
	if select_box=="SVM":
		p1 = prediction(model,SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm)
		score = model.score(X_train,Y_train)
	elif select_box == "Random Forest":
		p1 = prediction(modelRF,SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm)
		score = modelRF.score(X_train,Y_train)
	else:
		p1 = prediction(modelLR,SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm)
		score = modelLR.score(X_train,Y_train)

	st.write("Species Predicted :",p1)
	st.write("Accuracy: ",score)
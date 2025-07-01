import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import streamlit as st

#streamlit Configurations
st.set_page_config(page_title="Student Performance Predictor", layout="centered")
st.write ("This web app uses a simple machine learning model to predict a student's aveage score based on input subject marks")





training_data = pd.DataFrame({
    'maths' : np.random.randint(30,100,100),
    'science' :np.random.randint(30,100,100),
    'english':np.random.randint(30,100,100)
})

#Average for each student
training_data['average']=training_data.mean(axis=1)

X=training_data[['maths','science','english']]
y= training_data['average']

#Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Train Model
model = LinearRegression()
model.fit(X,y)

"""new_student= pd.DataFrame({
    'maths':[85],
    'science':[70],
    'english':[80]
})"""

#Input-slider
maths =st.slider("math scores", 0,100,50)
science =st.slider("science scores", 0,100,50)
english =st.slider("english scores", 0,100,50)

input_data=[[maths, science, english]]

#prediction = model.predict(new_student)
prediction = model.predict(input_data)[0]
st.success(f"Predicted score is :{round(prediction,2)}")

#Visualization

#Plotting training data and prediction
st.subheader("Visual Comparison")

fig, ax =plt.subplots()
ax.scatter(training_data['maths'],training_data['average'],color='blue', label="Training data")
ax.scatter(maths, prediction,color='red',label="Input", s=100)
ax.set_xlabel("Math score")
ax.set_ylabel("Predicted Average")
ax.set_title("Maths vs Average Prediction")
ax.legend()

#using streamlit
st.pyplot(fig)
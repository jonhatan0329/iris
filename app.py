import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Dictionary mapping numerical labels to flower names
flower_names = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}

#st.set_page_config(layout="wide")

st.title("Iris Prediction app")

# reading all the pickle files
model_scaler = pickle.load(open('scaler_iris.pkl','rb')) # 1st pickle scale model
model_lr_basic = pickle.load(open('model_lr_basic_iris.pkl','rb')) # 2nd pickle basic model
model_lr_smote= pickle.load(open('model_lr_smote_iris.pkl','rb')) # 3rd pickle smote model
model_smote_tuning= pickle.load(open('model_tuning_smote_iris.pkl','rb')) # 4th pickle smote tuning model
model_rfc= pickle.load(open('model_rfc_smote_iris.pkl','rb')) # 5th pickle rfc model

# user need to define the input
st.header("Enter the input values by User")

Sepal_L=st.slider(" Enter the float value for sepal length (cm)", 4.3,7.9)
#Sepal_L=st.number_input(" Enter the float value for sepal length (cm)")
Sepal_W=st.slider(" Enter the float value for sepal width (cm)", 2.0,4.4)
Petal_L=st.slider(" Enter the float value for petal length (cm)", 1.0,6.9)
Petal_W=st.slider(" Enter the float value for petal width (cm)", 0.1,2.5)

# create a dictionary for user_input
user_input={'sepal length (cm)': Sepal_L, 
            'sepal width (cm)': Sepal_W, 
            'petal length (cm)': Petal_L, 
            'petal width (cm)': Petal_W}

# convert to Dataframe
user_input_df=pd.DataFrame(user_input,index=[0])


# scale the user_data
user_input_df_scaled=model_scaler.transform(user_input_df)

st.write("Basic Model is simple logistic regression model using default parameters")

# user will select the model
selected_model=st.selectbox("Select one of the following models",("Basic Model","Smote Model","Smote Tuning Model","Random Forest Model"))
if st.button("Predict"):

    if selected_model=="Basic Model":
        prediction=model_lr_basic.predict(user_input_df_scaled)
        st.write("Basic Model is simple logistic regression model using default parameters")

    elif selected_model=="Smote Model":
        prediction=model_lr_smote.predict(user_input_df_scaled)
        
    elif selected_model=="Smote Tuning Model":
        prediction=model_smote_tuning.predict(user_input_df_scaled)
        
    elif selected_model=="Random Forest Model":
        prediction=model_rfc.predict(user_input_df_scaled)
        

# Ensure the model makes a prediction before accessing index [0]
    if model:
        prediction = model.predict(input_data)  # Get predicted class (0, 1, or 2)
        
        if len(prediction) > 0:  # Ensure prediction is not empty
            result = prediction[0]  # Get the first prediction
            flower_name = flower_names.get(result, "Unknown")  # Map number to name
            st.success(f"The predicted flower is: **{flower_name}** 🌸")
        else:
            st.error("Prediction failed. No result obtained.")
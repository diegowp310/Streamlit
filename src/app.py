from pickle import load
import streamlit as st

model = load(open("../models/decision_tree_classifier_default_42.sav", "rb"))
class_dict = {
    "0": "Iris setosa",
    "1": "Iris versicolor",
    "2": "Iris virginica"
}

st.title("Iris - Model prediction")

sepal_length = st.slider("Sepal length (cm)", min_value=4.0, max_value=8.0, value=5.8, step=0.1)
sepal_width = st.slider("Sepal width (cm)", min_value=2.0, max_value=4.5, value=3.0, step=0.1)
petal_length = st.slider("Petal length (cm)", min_value=1.0, max_value=7.0, value=3.8, step=0.1)
petal_width = st.slider("Petal width (cm)", min_value=0.1, max_value=2.5, value=1.2, step=0.1)

if st.button("Predict"):
    prediction = str(model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0])
    pred_class = class_dict[prediction]
    st.write("Prediction:", pred_class)

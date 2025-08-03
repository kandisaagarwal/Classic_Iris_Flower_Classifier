import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model("iris_nn_model.keras")
scaler = joblib.load("scaler.pkl")  # This is the full StandardScaler object

# Scale input using scaler's transform method
def scale_input(X):
    return scaler.transform(X)

st.title("🌼 Neural Network Iris Flower Classifier")

sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 5.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0)
petal_width = st.slider("Petal Width (cm)", 0.1, 3.0)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
scaled_input = scale_input(input_data)

try:
    prediction = model.predict(scaled_input)
    predicted_class = np.argmax(prediction)
    classes = ["setosa", "versicolor", "virginica"]
    st.write("Prediction raw output:", prediction)
except Exception as e:
    st.error(f"⚠️ Prediction failed: {e}")

# Predict

st.subheader("Prediction:")
st.success(f"This is a **{classes[predicted_class]}** 🌺")

# Neural Network Iris Flower Classifier

An interactive **Streamlit app** that classifies Iris flowers using a trained **TensorFlow Neural Network**. Users can adjust flower measurements with sliders and get real-time predictions.

This is my first attempt to implement my knowledge with Neurual Networks, and make a classic machine learning model. With the use of relu and softmax activation functions, and the use of Dense Neutron Layers, my model was able to acheive 96.67% accuracy. Due to limited complexity in this project, small dataset, and sparse features, working towards higher accuracy was a struggle. It was a fun project, my first ML project.

---

## 📌 Features

- Built with TensorFlow and Keras
- Input scaling using `StandardScaler`
- Live predictions with softmax output
- Streamlit-powered interactive UI
- Model saved in modern `.keras` format

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/iris-classifier.git
cd iris-classifier
```

### 2. Run the App

```bash
streamlit run app.py
```

---

## 🧠 How It Works

1. Users enter flower measurements using Streamlit sliders.

2. Inputs are scaled using a pre-trained StandardScaler (scaler.pkl).

3. A TensorFlow neural network (iris_nn_model.keras) predicts the flower class.

4. The predicted class is shown with a 🌺 emoji.

---

## Project Structure

```bash
├── app.py                 # Main Streamlit app
├── iris_nn_model.keras    # Trained TensorFlow model
├── scaler.pkl             # StandardScaler used to normalize inputs
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```
---

## Example Inputs

| Sepal Length | Sepal Width | Petal Length | Petal Width | Prediction |
| ------------ | ----------- | ------------ | ----------- | ---------- |
| 5.1 cm       | 3.5 cm      | 1.4 cm       | 0.2 cm      | Setosa     |
| 6.0 cm       | 3.0 cm      | 4.5 cm       | 1.5 cm      | Versicolor |
| 6.3 cm       | 3.3 cm      | 6.0 cm       | 2.5 cm      | Virginica  |

---

## 📚 Tech Stack

TensorFlow
scikit-learn
Streamlit
NumPy
joblib

---

## 🙌 Acknowledgments
Special thanks to the creators of the Iris dataset and the open-source ML community.


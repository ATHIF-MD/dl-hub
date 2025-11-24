# 🧠 DL Hub (Deep Learning Prediction Platform)

**DL Hub** is a specialized full-stack AI application focused on **Deep Learning** architectures. It demonstrates the end-to-end deployment of Neural Networks using **TensorFlow/Keras** for the model building and **FastAPI** for the backend server.

Unlike standard machine learning apps, this project handles complex unstructured data types, including **Images (Computer Vision)** and **Text sequences (NLP)**, alongside traditional tabular data.

---

## 📚 Deep Learning Architectures & Use Cases

### 1. Artificial Neural Network (ANN)
* **Architecture:** A Multi-Layer Perceptron (MLP) with Dense layers, utilizing ReLU activation for hidden layers and Sigmoid for binary output.
* **Project Problem:** **Diabetes Risk Diagnosis**
    * *Goal:* Predict the precise probability (0-100%) of a patient having diabetes based on clinical metrics.
    * *Input Features:* Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age.
    * *Preprocessing:* StandardScaler (Feature Scaling).
    * <img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/629bb93c-ee80-452f-be86-f9925d970791" />

### 2. Recurrent Neural Network (RNN - LSTM)
* **Architecture:** Utilizes **Long Short-Term Memory (LSTM)** units to process sequential data, solving the "vanishing gradient" problem found in standard RNNs. Includes Embedding layers for vectorization.
* **Project Problem:** **Sentiment Analysis**
    * *Goal:* Analyze user-submitted text reviews to determine emotional tone (Positive, Negative, or Neutral).
    * *Input:* Raw text string.
    * *Preprocessing:* Tokenization, Stopword Removal, Stemming, Sequence Padding.
    * <img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/8146adae-759c-423e-90d7-57eb8bbf83dc" />

### 3. Convolutional Neural Network (CNN)
* **Architecture:** A deep vision model using **Conv2D** layers for feature extraction (edges, shapes) and **MaxPooling** for dimensionality reduction, followed by a Dense classification head.
* **Project Problem:** **Image Classification (Cats vs. Dogs)**
    * *Goal:* Automatically classify an uploaded image file as either a Cat or a Dog.
    * *Input:* Image File (JPG/PNG).
    * *Preprocessing:* Rescaling (1/255), Resizing (150x150px), Batch processing.
    * <img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/09c05112-711a-493e-ad44-3c8508ec86df" />

---

## 🛠️ Tech Stack

* **Core AI:** TensorFlow, Keras, NumPy, Scikit-Learn, NLTK, Pillow (PIL).
* **Backend API:** Python, FastAPI, Uvicorn.
* **Frontend:** HTML5, CSS3 (Cyberpunk/Dark Theme), JavaScript (Async Fetch API).
* **Deployment:** Git LFS (Large File Storage) for managing heavy model weights.

---

## 🚀 Installation & Setup

Follow these steps to run the project locally.

```1. Clone the Repository
bash
git clone [https://github.com/ATHIF-MD/dl-hub.git](https://github.com/ATHIF-MD/dl-hub.git)
cd dl-hub

2. Set Up Environment
It is recommended to use a virtual environment to manage dependencies.

Bash

# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate

3. Install Dependencies
Install the required packages (including TensorFlow and FastAPI):

Bash

pip install fastapi uvicorn tensorflow numpy pandas nltk pillow python-multipart

4. Run the Application
Start the server (Running on Port 8001 to avoid conflicts with other apps):

Bash

python main.py
Alternatively: uvicorn main:app --port 8001 --reload

5. Access the Hub
Open your web browser and navigate to: https://www.google.com/search?q=http://127.0.0.1:8001

📂 Project Structure
Plaintext

dl-hub/
│
├── models/                # Trained Deep Learning Models
│   ├── diabetes_model.keras   # ANN Model
│   ├── rnn_model.keras        # RNN Model
│   ├── cnn_model.keras        # CNN Model (Heavy File)
│   ├── scaler.pkl             # Scaler for ANN
│   └── tokenizer.pickle       # Tokenizer for RNN
│
├── static/                # Assets
│   └── style.css          # Dark Theme Styling
│
├── templates/             # UI
│   └── index.html         # Main Dashboard
│
├── main.py                # FastAPI Server & Preprocessing Logic
├── README.md              # Documentation
└── .gitattributes         # Git LFS Configuration

Author: Athif MD


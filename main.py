import uvicorn
import pickle
import numpy as np
import re
import tensorflow as tf
from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import img_to_array
from io import BytesIO
from PIL import Image

# --- Text Processing Imports ---
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

app = FastAPI()

# --- Setup ---
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- Load Models ---
# 1. ANN: Diabetes
try:
    ann_model = tf.keras.models.load_model('models/diabetes_model.keras')
    ann_scaler = pickle.load(open('models/scaler.pkl', 'rb'))
    print("✅ ANN Model Loaded")
except:
    ann_model = None
    print("⚠️ ANN Model Missing (Diabetes)")

# 2. RNN: Sentiment
try:
    rnn_model = tf.keras.models.load_model('models/rnn_model.keras')
    with open('models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    print("✅ RNN Model Loaded")
except:
    rnn_model = None
    print("⚠️ RNN Model Missing (Sentiment)")

# 3. CNN: Image (Cats vs Dogs)
try:
    cnn_model = tf.keras.models.load_model('models/cnn_model.keras')
    print("✅ CNN Model Loaded")
except:
    cnn_model = None
    print("⚠️ CNN Model Missing (Cats/Dogs)")


# --- Helper Functions ---
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

def preprocess_image(image_bytes):
    # 1. Open Image
    img = Image.open(BytesIO(image_bytes))
    # 2. Resize to 150x150 (MUST match training!)
    img = img.resize((150, 150))
    # 3. Convert to Array
    img_array = img_to_array(img)
    # 4. Normalize (0-255 -> 0-1)
    img_array = img_array / 255.0
    # 5. Add Batch Dimension (1, 150, 150, 3)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- Routes ---
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 1. ANN Prediction
@app.post("/predict/ann")
async def predict_ann(request: Request):
    if not ann_model: return JSONResponse(content={"error": "Model not loaded"}, status_code=500)
    try:
        form_data = await request.form()
        features = [float(form_data[f]) for f in ['pregnancies','glucose','bp','skin','insulin','bmi','dpf','age']]
        scaled_features = ann_scaler.transform([features])
        prediction = ann_model.predict(scaled_features)[0][0]
        
        risk_percent = round(prediction * 100, 2)
        result = "High Risk ⚠️" if prediction > 0.5 else "Low Risk ✅"
        return JSONResponse(content={"result": result, "details": f"{risk_percent}% Probability"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# 2. RNN Prediction
@app.post("/predict/rnn")
async def predict_rnn(request: Request):
    if not rnn_model: return JSONResponse(content={"error": "Model not loaded"}, status_code=500)
    try:
        form_data = await request.form()
        raw_text = form_data.get('text_input')
        cleaned_text = preprocess_text(raw_text)
        seq = tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(seq, maxlen=50, padding='post')
        
        pred_probs = rnn_model.predict(padded)[0]
        class_idx = np.argmax(pred_probs)
        
        labels = {0: 'Negative 😠', 1: 'Neutral 😐', 2: 'Positive 😊'}
        sentiment = labels[class_idx]
        confidence = round(pred_probs[class_idx] * 100, 1)
        return JSONResponse(content={"result": sentiment, "details": f"Confidence: {confidence}%"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# 3. CNN Prediction
@app.post("/predict/cnn")
async def predict_cnn(file: UploadFile = File(...)):
    if not cnn_model: return JSONResponse(content={"error": "Model not loaded"}, status_code=500)
    try:
        # 1. Read File
        image_bytes = await file.read()
        
        # 2. Process
        processed_img = preprocess_image(image_bytes)
        
        # 3. Predict
        prediction = cnn_model.predict(processed_img)[0][0]
        
        # 4. Interpret (Sigmoid: <0.5 = Cat, >0.5 = Dog)
        # Based on your training class_indices {'cats': 0, 'dogs': 1}
        label = "Dog 🐶" if prediction > 0.5 else "Cat 🐱"
        confidence = prediction if prediction > 0.5 else 1 - prediction
        confidence_percent = round(confidence * 100, 2)
        
        return JSONResponse(content={"result": label, "details": f"{confidence_percent}% Confidence"})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
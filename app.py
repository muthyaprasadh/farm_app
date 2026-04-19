from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import boto3
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ✅ Load trained model
model = load_model("plant_model.h5")

# ✅ Class names (from your dataset)
class_names = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

# AWS clients
s3 = boto3.client('s3')
polly = boto3.client('polly')
translate = boto3.client('translate')
sns = boto3.client('sns')

BUCKET = "farm-disease-images"

# ---------- FUNCTIONS ----------

def preprocess_image(file):
    img = Image.open(file).convert('RGB')
    img = img.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def format_disease_name(name):
    return name.replace("_", " ").replace("__", " - ")

def predict_disease(file):
    img = preprocess_image(file)
    pred = model.predict(img)
    disease = class_names[np.argmax(pred)]
    return format_disease_name(disease)

def get_advice(disease):
    return f"Recommended treatment for {disease}: Use suitable pesticide, apply in early morning, follow safety precautions."

def text_to_speech(text):
    response = polly.synthesize_speech(
        Text=text,
        OutputFormat='mp3',
        VoiceId='Aditi'
    )

    file_path = "speech.mp3"
    with open(file_path, 'wb') as f:
        f.write(response['AudioStream'].read())

    return file_path

def translate_text(text):
    result = translate.translate_text(
        Text=text,
        SourceLanguageCode='en',
        TargetLanguageCode='te'
    )
    return result['TranslatedText']

def send_sms(message):
    sns.publish(
        PhoneNumber='+919908657967',  # change if needed
        Message=message
    )

# ---------- API ----------

@app.route('/')
def home():
    return "🌱 Farm Disease Detection API Running"

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']

    # Upload to S3
    s3.upload_fileobj(file, BUCKET, file.filename)
    file.seek(0)  # reset file pointer

    # AI prediction
    disease = predict_disease(file)
    advice = get_advice(disease)

    # AWS features
    translated = translate_text(advice)
    audio = text_to_speech(advice)

    send_sms(f"Disease detected: {disease}")

    return jsonify({
        "disease": disease,
        "advice": advice,
        "translated": translated,
        "audio": audio
    })

# ---------- RUN ----------

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
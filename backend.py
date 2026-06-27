import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, jsonify
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'D:/cartoon_detection/uploads'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB max

# Settings
IMG_SIZE = 48
SEQUENCE_LENGTH = 10
CLASSES = ["Gun Violence", "Knife Violence", "Non-Violent"]
MODEL_PATH = "D:/cartoon_detection/model/violence_model.h5"

# Allowed video types
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Load model
print("Model loading...")
model = load_model(MODEL_PATH)
print("Model loaded! ")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_frames_from_video(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // SEQUENCE_LENGTH)

    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = frame / 255.0
        frames.append(frame)
        if len(frames) == SEQUENCE_LENGTH:
            break

    cap.release()

    while len(frames) < SEQUENCE_LENGTH:
        frames.append(frames[-1] if frames else np.zeros((IMG_SIZE, IMG_SIZE, 3)))

    return np.array(frames)

def predict_video(video_path):
    frames = extract_frames_from_video(video_path)
    frames = np.expand_dims(frames, axis=0)
    prediction = model.predict(frames)
    class_idx = np.argmax(prediction[0])
    confidence = float(prediction[0][class_idx] * 100)
    return CLASSES[class_idx], confidence, prediction[0].tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return render_template('index.html', error="No Video !")

    file = request.files['video']

    if file.filename == '':
        return render_template('index.html', error="choose Video !")

    if not allowed_file(file.filename):
        return render_template('index.html', 
            error="Type of file is wrong ! only  MP4, AVI, MOV, MKV Accepted.")

    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(video_path)

    result, confidence, all_preds = predict_video(video_path)

    return render_template('result.html',
        result=result,
        confidence=f"{confidence:.2f}",
        filename=filename,
        gun_conf=f"{all_preds[0]*100:.2f}",
        knife_conf=f"{all_preds[1]*100:.2f}",
        nonviolent_conf=f"{all_preds[2]*100:.2f}"
    )

if __name__ == '__main__':
    app.run(debug=True)
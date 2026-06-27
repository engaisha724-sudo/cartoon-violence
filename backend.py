import os
import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

IMG_SIZE = 48
SEQUENCE_LENGTH = 10
CLASSES = ["Gun Violence", "Knife Violence", "Non-Violent"]
MODEL_PATH = "model/violence_model.onnx"
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

print("Model loading...")
session = ort.InferenceSession(MODEL_PATH)
print("Model loaded! ")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
    return np.array(frames, dtype=np.float32)

def predict_video(video_path):
    frames = extract_frames_from_video(video_path)
    frames = np.expand_dims(frames, axis=0)
    input_name = session.get_inputs()[0].name
    prediction = session.run(None, {input_name: frames})[0]
    class_idx = np.argmax(prediction[0])
    confidence = float(prediction[0][class_idx] * 100)
    return CLASSES[class_idx], confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return render_template('index.html', error="no video!")
    file = request.files['video']
    if file.filename == '':
        return render_template('index.html', error="choose a Video !")
    if not allowed_file(file.filename):
        return render_template('index.html', error="only  MP4, AVI, MOV, MKV files are allowed .")
    filename = secure_filename(file.filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(video_path)
    result, confidence = predict_video(video_path)
    return render_template('result.html', result=result, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Settings
IMG_SIZE = 48
SEQUENCE_LENGTH = 10
FRAMES_PATH = "D:/cartoon_detection/dataset/frames"
CLASSES = ["gun", "knife", "non_violent"]
MODEL_SAVE_PATH = "D:/cartoon_detection/model/violence_model.h5"

def load_frames_sequence(folder_path):
    frames = []
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".jpg")])
    if not files:
        return None
    step = max(1, len(files) // SEQUENCE_LENGTH)
    selected = files[::step][:SEQUENCE_LENGTH]
    for f in selected:
        img = cv2.imread(os.path.join(folder_path, f))
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        frames.append(img)
    while len(frames) < SEQUENCE_LENGTH:
        if frames:
            frames.append(frames[-1])
        else:
            frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3)))
    return np.array(frames)

def load_dataset():
    X, y = [], []
    for label_idx, label in enumerate(CLASSES):
        label_path = os.path.join(FRAMES_PATH, label)
        if not os.path.exists(label_path):
            print(f"Ma jirto: {label_path}")
            continue
        folders = [f for f in os.listdir(label_path)
                   if os.path.isdir(os.path.join(label_path, f))]
        print(f"\n[{label.upper()}] — {len(folders)} videos")
        for folder in folders:
            folder_path = os.path.join(label_path, folder)
            sequence = load_frames_sequence(folder_path)
            if sequence is not None:
                X.append(sequence)
                y.append(label_idx)
    return np.array(X, dtype=np.float32), np.array(y)

def build_model():
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False
    cnn = Sequential([
        base_model,
        GlobalAveragePooling2D()
    ])
    model = Sequential([
        TimeDistributed(cnn, input_shape=(SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3)),
        LSTM(32, return_sequences=False),
        Dropout(0.5),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(len(CLASSES), activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

print("Dataset loading...")
X, y = load_dataset()
print(f"\n total samples: {len(X)}")

y_cat = to_categorical(y, num_classes=len(CLASSES))
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

print("\nModel building...")
model = build_model()
model.summary()

callbacks = [
    ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, verbose=1),
    EarlyStopping(patience=10, restore_best_weights=True, verbose=1)
]

print("\n Beganing Training...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=4,
    validation_data=(X_test, y_test),
    callbacks=callbacks
)

loss, acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {acc*100:.2f}%")
print(f"Model saved: {MODEL_SAVE_PATH}")
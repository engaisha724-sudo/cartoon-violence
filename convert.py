import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import tf2onnx
import tensorflow as tf

model = tf.keras.models.load_model('model/violence_model.h5', compile=False)
tf2onnx.convert.from_keras(model, output_path='model/violence_model.onnx')
print(" Done!")
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("tensorflow_model.h5")

# Convert to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open("tensorflow_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model successfully converted to TFLite format!")
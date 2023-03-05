import streamlit as st
import numpy as np
import tensorflow as tf

# Load TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="InceptionResNetV2Skripsi.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define a function to resize the input image
def resize_image(image):
    # Resize the image to 150x150 pixels
    resized_image = tf.image.resize(image, [150, 150])
    return resized_image.numpy()

# Define a function to run inference on the TensorFlow Lite model
def classify_image(image):
    # Pre-process the input image
    resized_image = resize_image(image)
    input_data = np.expand_dims(resized_image, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the output probabilities
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]

# Define the labels for the 7 classes
labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

from PIL import Image
# Define the main Streamlit app
def main():
    st.title("Skin Cancer Classification")

    # Get the input image from the user
    image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # Show the input image
    if image is not None:
        image = np.array(Image.open(image))
        st.image(image, width=150)

        # Run inference on the input image
        probs = classify_image(image)

        # Display the top 3 predictions
        top_3_indices = np.argsort(probs)[::-1][:3]
        st.write("Top 3 predictions:")
        for i in range(3):
            st.write("%d. %s (%.2f%%)" % (i + 1, labels[top_3_indices[i]], probs[top_3_indices[i]] * 100))

if __name__ == '__main__':
    main()

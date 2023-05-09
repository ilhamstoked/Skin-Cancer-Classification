import streamlit as st
import numpy as np
import tensorflow as tf
import time
from tcp_latency import measure_latency

# Measure Latency

#35.201.127.49:443
#192.168.18.6:8501

# Load TensorFlow Lite model
start_time = time.time()
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
    with st.spinner('Classifying...'):
        interpreter.invoke()


    # Get the output probabilities
    output_data = interpreter.get_tensor(output_details[0]['index'])

    return output_data[0], classifying_duration

# Define the labels for the 7 classes
labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

from PIL import Image
# Define the main Streamlit app
def main():
    st.title("Skin Cancer Classification")

    st.write("The application allows users to quickly and easily check for signs of skin cancer from the comfort of their own homes. It's important to note that the application is not a substitute for medical advice, but rather a tool to help users identify potential skin issues and seek professional help if necessary. Heres the overview:")
    st.write("1.	The user is prompted to upload an image of a skin lesion.")
    st.write("2.	Once the user uploads an image, it is displayed on the screen and the application runs the image through a machine learning model.")
    st.write("3.	The model outputs the top 3 predictions for the type of skin cancer (out of 7 classes) that the lesion may be, along with the confidence level for each prediction.")
    st.write("4.	The user is shown the top 3 predictions and their confidence levels.")
    st.write("5.	If the model's confidence level is less than 70%, the application informs the user that the skin is either healthy or the input image is unclear. It is also recommended that the user consults a dermatologist for any skin concerns.")
    st.write ("6.	The application also measures the latency or response time for the model to classify the image, and displays it to the user.")
    st.write("Please note that this model still has room for academic revision as it can only classify the following 7 classes")
    st.write("- ['akiec'](https://en.wikipedia.org/wiki/Squamous-cell_carcinoma) - squamous cell carcinoma (actinic keratoses dan intraepithelial carcinoma),")
    st.write("- ['bcc'](https://en.wikipedia.org/wiki/Basal-cell_carcinoma) - basal cell carcinoma,")
    st.write("- ['bkl'](https://en.wikipedia.org/wiki/Seborrheic_keratosis) - benign keratosis (serborrheic keratosis),")
    st.write("- ['df'](https://en.wikipedia.org/wiki/Dermatofibroma) - dermatofibroma, ")
    st.write("- ['nv'](https://en.wikipedia.org/wiki/Melanocytic_nevus) - melanocytic nevus, ")
    st.write("- ['mel'](https://en.wikipedia.org/wiki/Melanoma) - melanoma,")
    st.write("- ['vasc'](https://en.wikipedia.org/wiki/Vascular_anomaly) - vascular skin (Cherry Angiomas, Angiokeratomas, Pyogenic Granulomas.)")
    st.write("Due to imperfection of the model and a room of improvement for the future, if the probabilities shown are less than 70%, the skin is either healthy or the input image is unclear. This means that the model can be the first diagnostic of your skin illness. As precautions for your skin illness, it is better to do consultation with dermatologist. ")
    
    
    # Get the input image from the user
    image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # Show the input image
    if image is not None:
        image = np.array(Image.open(image).convert("RGB"))
        st.image(image, width=150)

        # Run inference on the input image
        probs, classifying_duration = classify_image(image)

        # Display the classification duration
        st.write(f"Classification duration: {classifying_duration:.4f} seconds")

        # Display the top 3 predictions
        top_3_indices = np.argsort(probs)[::-1][:3]
        st.write("Top 3 predictions:")
        for i in range(3):
            st.write("%d. %s (%.2f%%)" % (i + 1, labels[top_3_indices[i]], probs[top_3_indices[i]] * 100))
            
        latency = measure_latency(host='35.201.127.49', port=443)
        st.write("Network Latency:")
        st.write(latency[0])
        end_time = time.time()

    # Calculate the classification duration
    classifying_duration = end_time - start_time
    
if __name__ == '__main__':
    main()

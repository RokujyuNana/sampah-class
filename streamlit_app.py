import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path='sampah_model.tflite')
interpreter.allocate_tensors()

# Get input and output details of the model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape'][1:3]  # Exclude the batch dimension

# Define the garbage categories
categories = ['Battery', 'Biological', 'Cardboard', 'Clothes', 'Glass', 
              'Metal', 'Paper', 'Plastic', 'Shoes', 'Trash']

def predict_image(img):
    # Convert the image to RGB if it is not already
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize the image to match the input shape of the model
    img = img.resize(input_shape)

    # Convert the image to an array and normalize it
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Set the tensor to point to the input data
    interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
    
    # Run the interpreter
    interpreter.invoke()
    
    # Get the prediction
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = categories[np.argmax(output_data)]
    
    return prediction

# Streamlit interface
st.title('Garbage Classification')
st.write("""
### Garbage Categories:
1. **Battery**: Hazardous waste like batteries.
2. **Biological**: Organic waste, such as food scraps.
3. **Cardboard**: Recyclable cardboard materials.
4. **Clothes**: Textile waste like old clothes.
5. **Glass**: Glass waste.
6. **Metal**: Metal objects like cans or tools.
7. **Paper**: Paper waste, such as newspapers.
8. **Plastic**: Plastic materials like bottles.
9. **Shoes**: Footwear waste.
10. **Trash**: General non-recyclable waste.
""")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    if st.button('Predict'):
        prediction = predict_image(img)
        st.write(f"Prediction: {prediction}")

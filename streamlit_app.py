import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the garbage classification model
model = load_model('garbage_classification_model.h5')

def predict_image(img):
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make a prediction
    predictions = model.predict(img_array)
    
    # Define categories
    categories = ['Battery', 'Biological', 'Brown-Glass', 'Cardboard', 'Clothes', 'Green-Glass', 
                  'Metal', 'Paper', 'Plastic', 'Shoes', 'Trash', 'White-Glass']
    
    return categories[np.argmax(predictions)]

# Streamlit interface
st.title('Garbage Classification')
st.write("""
### Garbage Categories:
1. **Battery**: Hazardous waste like batteries.
2. **Biological**: Organic waste, such as food scraps.
3. **Brown-Glass**: Brown-colored glass waste.
4. **Cardboard**: Recyclable cardboard materials.
5. **Clothes**: Textile waste like old clothes.
6. **Green-Glass**: Green-colored glass waste.
7. **Metal**: Metal objects like cans or tools.
8. **Paper**: Paper waste, such as newspapers.
9. **Plastic**: Plastic materials like bottles.
10. **Shoes**: Footwear waste.
11. **Trash**: General non-recyclable waste.
12. **White-Glass**: White or clear glass waste.
""")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img = img.resize((150, 150))
    st.image(img, caption='Uploaded Image', use_column_width=True)

    if st.button('Predict'):
        prediction = predict_image(img)
        st.write(f"Prediction: {prediction}")

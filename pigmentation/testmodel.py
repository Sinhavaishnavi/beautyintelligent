import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('pigmentation_detector.h5')

# Function to process the image and make a prediction
def predict_skin_condition(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(128,128))  # Resize to match model input size
    img_array = image.img_to_array(img)  # Convert the image to a numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add an extra dimension for batch size (1 image)
    img_array = img_array / 255.0  # Normalize the image

    # Make a prediction
    prediction = model.predict(img_array)
    
    # Map the prediction to "Clear Skin" or "Pigmented Skin"
    if prediction[0] > 0.5:
        result = 'Pigmented Skin'
    else:
        result = 'Clear Skin'

    # Display the image and prediction
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Prediction: {result}")
    plt.show()

    return result

# Test the function with an image path
img_path = r'C:\Users\LENOVO\OneDrive\Desktop\beautyintelligent\images\oily.jpeg' # Replace this with the path to your image
skin_condition = predict_skin_condition(img_path)
print(f"The predicted skin condition is: {skin_condition}")

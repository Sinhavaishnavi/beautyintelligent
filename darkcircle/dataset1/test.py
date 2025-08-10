import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('dark_detector.h5')

# Function to process the image and make a prediction
def predict_dark_circles(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(128, 128))  # Resize to match model input size
    img_array = image.img_to_array(img)  # Convert the image to a numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add an extra dimension for batch size (1 image)
    img_array = img_array / 255.0  # Normalize the image

    # Make a prediction
    prediction = model.predict(img_array)
    
    # Map the prediction to "Dark Circles" or "No Dark Circles"
    if prediction[0] > 0.5:
        result = 'Dark Circles'
    else:
        result = 'No Dark Circles'

    # Display the image and prediction
    img_for_display = image.load_img(img_path)  # Load original image for display
    plt.imshow(img_for_display)
    plt.axis('off')
    plt.title(f"Prediction: {result}")
    plt.show()

    return result

# Test the function with an image path
img_path = r'C:\Users\LENOVO\OneDrive\Desktop\beautyintelligent\images\download.jpeg'  # Replace this with the path to your image
dark_circle_condition = predict_dark_circles(img_path)
print(f"The predicted condition is: {dark_circle_condition}")

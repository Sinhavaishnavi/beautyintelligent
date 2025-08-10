import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained model for dark circle detection
dark_circle_model = load_model('dark_detector.h5')  # Replace with your model file

# Function to process the image and make a prediction
def predict_dark_circle_condition(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(128 , 128))  # Resize to match model input size
    img_array = image.img_to_array(img)  # Convert the image to a numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add an extra dimension for batch size (1 image)
    img_array = img_array / 255.0  # Normalize the image

    # Make a prediction
    prediction = dark_circle_model.predict(img_array)
    
    # Map the prediction to "No Dark Circles" or "Dark Circles Present"
    if prediction[0] > 0.5:
        result = 'Dark Circles Present'
    else:
        result = 'No Dark Circles'

    # Display the image and prediction
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Prediction: {result}")
    plt.show()

    return result

# Test the function with an image path
dark_circle_img_path = r'C:\Users\LENOVO\OneDrive\Desktop\beautyintelligent\darkcircle\dataset1\train\darkcircletrain\_1562650870001_jpg.rf.fa51a05e59a23b0e886d4e1c155b7677.jpg'  # Replace with the path to your image
dark_circle_condition = predict_dark_circle_condition(dark_circle_img_path)
print(f"The predicted dark circle condition is: {dark_circle_condition}")

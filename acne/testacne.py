import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained model for acne detection
acne_model = load_model('acne1_detector_mobilenetv2.h5')

# Function to process the image and make a prediction
def predict_acne_condition(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224,224))  # Resize to match model input size
    img_array = image.img_to_array(img)  # Convert the image to a numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add an extra dimension for batch size (1 image)
    img_array = img_array / 255.0  # Normalize the image

    # Make a prediction
    prediction = acne_model.predict(img_array)
    
    # Map the prediction to "No Acne" or "Acne Present"
    if prediction[0] > 0.5:
        result = ' NO  Acne Present'
    else:
        result = ' Acne'

    # Display the image and prediction
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Prediction: {result}")
    plt.show()

    return result

# Test the function with an image path
acne_img_path = r'C:\Users\LENOVO\OneDrive\Desktop\beautyintelligent\acne\acne\valid\acne\berjerawat (102).jpg'  # Replace this with the path to your image
acne_condition = predict_acne_condition(acne_img_path)
print(f"The predicted acne condition is: {acne_condition}")

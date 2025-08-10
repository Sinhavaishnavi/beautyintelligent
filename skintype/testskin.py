import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('skin_type_classifier.h5')  # Replace with your model's filename

# Function to process the image and make a prediction
def predict_skin_type(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224,224))  # Resize to match model input size
    img_array = image.img_to_array(img)  # Convert the image to a numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add an extra dimension for batch size (1 image)
    img_array = img_array / 255.0  # Normalize the image

    # Make a prediction
    prediction = model.predict(img_array)
    
    # Map the prediction to skin types
    skin_types = ['Dry', 'Normal', 'Oily']
    predicted_class = np.argmax(prediction)  # Get the index of the highest probability
    result = skin_types[predicted_class]

    # Display the image and prediction
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Prediction: {result}")
    plt.show()

    return result

# Test the function with an image path
img_path = r'C:\Users\LENOVO\OneDrive\Desktop\beautyintelligent\skintype\dataset\train\dry\dry_0b973f46a06123d014f1_jpg.rf.d82b9f0684e07553d77c31a88380a7ba.jpg'  # Replace this with the path to your image
skin_type = predict_skin_type(img_path)
print(f"The predicted skin type is: {skin_type}")
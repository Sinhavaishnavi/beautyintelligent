🌸 Intelligent Beauty Advisor

An interactive web application built with Streamlit that uses deep learning models to analyze users' skin for conditions like acne, dark circles, and pigmentation. Based on the analysis, it provides personalized skincare recommendations.
DEPLOY -https://beautyintelligent-bsquorloswmedif2s8rscq.streamlit.app/
<img width="1910" height="824" alt="Screenshot 2025-08-14 230617" src="https://github.com/user-attachments/assets/a8757093-8e44-4732-aef3-43b0a3a66fa0" />

<img width="1917" height="859" alt="Screenshot 2025-08-14 230652" src="https://github.com/user-attachments/assets/1b946d45-a79e-4953-84b4-6d486833f0ec" />
<img width="809" height="848" alt="Screenshot 2025-08-14 230712" src="https://github.com/user-attachments/assets/107c98fa-0841-41d9-bbd5-e056880e3e10" />

✨ Key Features
User Authentication: Secure login and registration system to manage user data and history.

AI-Powered Skin Analysis: Upload an image or use your webcam to get an instant analysis of your skin. The app detects:

Skin Type (Oily, Dry, Normal)

Presence of Acne

Presence of Dark Circles

Presence of Pigmentation

Personalized Recommendations: Receive tailored skincare advice, including routine tips and key ingredients based on the analysis results.

Analysis History: Registered users can save their analysis results to track their skin's progress over time.

Interactive & User-Friendly Interface: A beautiful and responsive UI built with a custom color theme and smooth animations for a great user experience.

🛠️ Technologies Used
Framework: Streamlit

Deep Learning: TensorFlow & Keras

Image Processing: OpenCV, Pillow (PIL)

Data Handling: Pandas, NumPy

Database: SQLite for user and history management.

🚀 Setup and Installation
To run this project locally, follow these steps:

1. Clone the Repository

git clone https://github.com/Sinhavaishnavi/beautyintelligent.git
cd beautyintelligent

2. Create a Virtual Environment (Recommended)

# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

3. Install Dependencies

pip install -r requirements.txt



4. Download the Pre-trained Models

You will need the following .h5 model files in the root directory of the project:

acne1_detector_mobilenetv2.h5

dark_detector.h5

pigmentation_detector.h5

final_skin_type_model_resnet.h5

Make sure these model files are available in your project folder before running the app.

5. Run the Streamlit Application

streamlit run appp.py 



Creating the requirements.txt File
If you don't have a requirements.txt file, you can create one with the following content. These are the essential libraries needed to run the application.

streamlit
tensorflow
pandas
numpy
opencv-python-headless
pillow

📖 How to Use the Application
Register/Login: Create a new account or log in with existing credentials.

Navigate to Skin Analysis: Use the sidebar to go to the "Skin Analysis" page.

Provide an Image:

Click "Upload an Image" to select a photo from your device.

Or, click "Use Webcam" to take a live picture.

Analyze: Once the image is displayed, click the "Analyze Image" button.

View Results: The app will show the analysis results and personalized recommendations in expandable sections.

Save & Track: You can save the results to your profile and view your entire history on the "View History" page.

🤝 Contributing
Contributions are welcome! If you have suggestions for improvements or want to add new features, feel free to fork the repository and submit a pull request.

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request

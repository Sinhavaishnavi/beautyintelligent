import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import cv2
import os
import sqlite3
import hashlib
import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="Intelligent Beauty Advisor",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS with New Pink Shade, Animations, and Refined Theme ---
st.markdown("""
<style>
/* Add a fade-in animation */
@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
  [theme]
primaryColor="#6d282a"
backgroundColor="#b78097"
secondaryBackgroundColor="#23252f"
textColor="#605C5C"

}

/* Main background with the new pink shade */
body, .stApp {
    background-color: #FFDCDC; /* New Dusty Rose/Pink Shade */
    color: #382e2c; /* A softer black for text */
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    animation: fadeIn 0.8s ease-in-out; /* Apply animation */
}
/* Headers */
h1, h2, h3, h4, h5, h6 {
    color: #382e2c;
    font-weight: 600;
}
/* Buttons - Adjusted for the new theme */
.stButton > button {
    background-color: #7C444F; /* Using Cabaret for a strong but fitting accent */
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 24px;
    font-weight: 600;
    transition: background-color 0.3s ease, transform 0.2s ease;
}
.stButton > button:hover {
    background-color: #A93150B; /* A darker shade for hover */
    transform: scale(1.03); /* Add a slight zoom on hover */
}
/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #FFF2EB; /* French Grey - still works well */
    color: #382e2c;
}
/* Inputs */
.stTextInput input, .stSelectbox select, .stTextArea textarea {
    border-radius: 6px;
    border: 1.5px solid #9F5255;
    padding: 6px 8px;
    font-size: 1rem;
    color: #382e2c;
    background-color: #FFFFFF;
    text-color: black;
}
/* File uploader */
.stFileUploader > div {
    border: 2px dashed #9F5255;
    border-radius: 8px;
    padding: 16px;
    background-color: rgba(255, 255, 255, 0.5);
}
/* Metric labels and values */
[data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
    color: #382e2c;
}
/* Expander for recommendations */
.st-expander {
    border-color: #9F5255 !important;
    transition: box-shadow 0.3s ease;
}
.st-expander:hover {
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)


# --- Database Setup ---
DB_NAME = "skin_analysis.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            acne TEXT,
            dark_circles TEXT,
            pigmentation TEXT,
            skin_type TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# --- Model Loading with caching ---
class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, **kwargs):
        if 'groups' in kwargs:
            del kwargs['groups']
        super().__init__(**kwargs)

@st.cache_resource
def load_all_models():
    models = {}
    model_paths = {
        "acne": "acne1_detector_mobilenetv2.h5",
        "dark": "dark_detector.h5",
        "pigment": "pigmentation_detector.h5",
        "skin": "final_skin_type_model_resnet.h5"
    }
    custom_objects = {'DepthwiseConv2D': CustomDepthwiseConv2D}
    for name, path in model_paths.items():
        if os.path.exists(path):
            try:
                models[name] = load_model(path, custom_objects=custom_objects, compile=False)
            except Exception as e:
                st.error(f"Error loading {name} model: {e}")
                models[name] = None
        else:
            st.warning(f"Model file not found: {path}")
            models[name] = None
    return models

# --- Image preprocessing and prediction ---
def preprocess_frame(frame, target_size):
    img = cv2.resize(frame, target_size)
    img_array = img_to_array(img)
    img_array = img_array.astype("float32") / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_all_conditions(image, models):
    frame = np.array(image.convert('RGB'))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    img_224 = preprocess_frame(frame, (224, 224))
    img_128 = preprocess_frame(frame, (128, 128))
    predictions = {}
    skin_labels = ['Dry', 'Normal', 'Oily']
    if models.get("acne"):
        predictions["Acne"] = "Yes" if models["acne"].predict(img_224)[0][0] > 0.5 else "No"
    if models.get("dark"):
        predictions["Dark Circles"] = "Yes" if models["dark"].predict(img_128)[0][0] > 0.5 else "No"
    if models.get("pigment"):
        predictions["Pigmentation"] = "Yes" if models["pigment"].predict(img_128)[0][0] > 0.5 else "No"
    if models.get("skin"):
        predictions["Skin Type"] = skin_labels[np.argmax(models["skin"].predict(img_224))]
    return predictions

# --- Recommendation System ---
def get_recommendations(predictions):
    recs = {}
    skin_type = predictions.get("Skin Type")
    if skin_type == 'Oily':
        recs['Skin Type'] = "**Routine:** Use a gel/foam cleanser and a lightweight, oil-free moisturizer. \n\n**Ingredients:** Look for salicylic acid, niacinamide, and clay."
    elif skin_type == 'Dry':
        recs['Skin Type'] = "**Routine:** Use a creamy cleanser and a rich moisturizer. Consider a hydrating serum. \n\n**Ingredients:** Hyaluronic acid, ceramides, and glycerin are great."
    elif skin_type == 'Normal':
        recs['Skin Type'] = "**Routine:** Maintain balance with a gentle cleanser and moisturizer. \n\n**Key Tip:** Daily broad-spectrum sunscreen is crucial."
    if predictions.get("Acne") == 'Yes':
        recs['Acne'] = "**Treatment:** Use products with salicylic acid or benzoyl peroxide. \n\n**Habits:** Avoid touching your face and use non-comedogenic makeup."
    if predictions.get("Dark Circles") == 'Yes':
        recs['Dark Circles'] = "**Treatment:** An eye cream with caffeine, Vitamin C, or retinol can help. \n\n**Lifestyle:** Aim for 7-9 hours of sleep and stay hydrated."
    if predictions.get("Pigmentation") == 'Yes':
        recs['Pigmentation'] = "**Protection:** SPF 30+ sunscreen daily is your most important tool. \n\n**Treatment:** Incorporate Vitamin C, azelaic acid, or niacinamide."
    return recs

# --- Helper function for navigation ---
def navigate_to(page_name):
    # Store the current page as the previous one, unless we are already on the target page
    if 'page' in st.session_state and st.session_state.page != page_name:
        st.session_state.previous_page = st.session_state.page
    st.session_state.page = page_name

# --- Pages ---
def login_page():
    st.header("Login to Your Account")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            cursor.execute("SELECT id, password_hash FROM users WHERE username = ?", (username,))
            user_record = cursor.fetchone()
            conn.close()
            if user_record and user_record[1] == hash_password(password):
                st.session_state['logged_in'] = True
                st.session_state['user_id'] = user_record[0]
                st.session_state['username'] = username
                navigate_to('Analysis') # Use navigation function
                # REMOVED st.experimental_rerun() to fix double-click issue
            else:
                st.error("Invalid username or password.")

def register_page():
    st.header("Create a New Account")
    with st.form("register_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Register")
        if submitted:
            if not username or not password:
                st.error("Please fill all fields.")
            else:
                conn = sqlite3.connect(DB_NAME)
                try:
                    conn.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, hash_password(password)))
                    conn.commit()
                    st.success("Registration successful! Please login.")
                    navigate_to('Login')
                except sqlite3.IntegrityError:
                    st.error("Username already exists.")
                finally:
                    conn.close()

def analysis_page(models):
    st.title("🔬 Skin Analysis & Recommendations")
    # NEW: Add a back button if there's a previous page in history
    if 'previous_page' in st.session_state:
        if st.button("⬅️ Go Back"):
            navigate_to(st.session_state.previous_page)
            st.rerun() # Use st.rerun() here to force immediate page switch
    st.write("Upload an image or use your webcam to analyze your skin and get personalized advice.")
    col1, col2 = st.columns([1,1])
    with col1:
        st.subheader("Image Input")
        source = st.radio("Choose your image source:", ["Upload an Image", "Use Webcam"], horizontal=True)
        image = None
        if source == "Upload an Image":
            if uploaded_file := st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"]):
                image = Image.open(uploaded_file)
        else:
            if camera_input := st.camera_input("Take a picture"):
                image = Image.open(camera_input)
        if image:
            st.image(image, caption="Your Image", use_column_width=True)
            if st.button("Analyze Image", use_container_width=True, key="analyze"):
                with st.spinner("Analyzing..."):
                    st.session_state.last_analysis = predict_all_conditions(image, models)
                    st.session_state.last_recommendations = get_recommendations(st.session_state.last_analysis)
    with col2:
        st.subheader("Analysis & Advice")
        if 'last_analysis' in st.session_state:
            st.markdown("#### Results")
            results = st.session_state.last_analysis
            c1, c2 = st.columns(2)
            c1.metric("Skin Type", results.get("Skin Type", "N/A"))
            c2.metric("Acne", results.get("Acne", "N/A"))
            c1.metric("Dark Circles", results.get("Dark Circles", "N/A"))
            c2.metric("Pigmentation", results.get("Pigmentation", "N/A"))
            if st.button("Save Results", use_container_width=True):
                # Save logic remains the same
                conn = sqlite3.connect(DB_NAME)
                conn.execute('INSERT INTO analysis_history (user_id, timestamp, acne, dark_circles, pigmentation, skin_type) VALUES (?, ?, ?, ?, ?, ?)',
                             (st.session_state.user_id, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                              results.get("Acne"), results.get("Dark Circles"), results.get("Pigmentation"), results.get("Skin Type")))
                conn.commit()
                conn.close()
                st.success("Analysis results saved successfully!")
            st.markdown("---")
            st.markdown("#### Recommendations")
            if recommendations := st.session_state.get('last_recommendations', {}):
                for concern, advice in recommendations.items():
                    with st.expander(f"✨ Advice for {concern}"):
                        st.markdown(advice)
            else:
                st.info("No specific recommendations needed based on the analysis.")
        else:
            st.info("Please provide an image and click 'Analyze Image' to get results and recommendations.")

def history_page():
    st.title("📜 Your Analysis History")
    # NEW: Add a back button
    if 'previous_page' in st.session_state:
        if st.button("⬅️ Go Back"):
            navigate_to(st.session_state.previous_page)
            st.rerun() # Force immediate page switch
    user_id = st.session_state.get('user_id')
    if not user_id: return
    conn = sqlite3.connect(DB_NAME)
    try:
        df = pd.read_sql_query("SELECT timestamp AS 'Date & Time', acne AS Acne, dark_circles AS 'Dark Circles', pigmentation AS Pigmentation, skin_type AS 'Skin Type' FROM analysis_history WHERE user_id = ? ORDER BY timestamp DESC", conn, params=(user_id,))
        if not df.empty:
            st.dataframe(df, use_container_width=True)
        else:
            st.info("You have no saved analysis history.")
    finally:
        conn.close()

# --- Main App Router ---
def main():
    init_db()
    models = load_all_models()

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.page = 'Login'

    # Sidebar navigation
    st.sidebar.title("Navigation")
    if st.session_state.logged_in:
        st.sidebar.success(f"Logged in as {st.session_state.username}")
        if st.sidebar.button("Skin Analysis", use_container_width=True):
            navigate_to('Analysis')
        if st.sidebar.button("View History", use_container_width=True):
            navigate_to('History')
        if st.sidebar.button("Logout", use_container_width=True):
            # Clear session state on logout
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state.logged_in = False
            st.session_state.page = 'Login'
    else:
        if st.sidebar.button("Login", use_container_width=True):
            navigate_to('Login')
        if st.sidebar.button("Register", use_container_width=True):
            navigate_to('Register')

    # Page content based on state
    # This part runs after the sidebar, so the state is already set by button clicks
    page = st.session_state.get('page', 'Login')
    if page == 'Login':
        login_page()
    elif page == 'Register':
        register_page()
    elif page == 'Analysis' and st.session_state.logged_in:
        analysis_page(models)
    elif page == 'History' and st.session_state.logged_in:
        history_page()
    else:
        # Fallback to login if trying to access a protected page while not logged in
        st.warning("Please log in to continue.")
        login_page()

if __name__ == '__main__':
    main()
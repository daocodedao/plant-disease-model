import streamlit as st
import requests
from PIL import Image
import io
import base64
import json

# -------------------------------------
# 📌 Streamlit Page Configuration
# -------------------------------------
st.set_page_config(
    page_title="Plant Disease Recognition System",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------
# 📌 Custom CSS for UI improvement
# -------------------------------------
st.markdown("""
    <style>
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .stButton>button { background-color: #4CAF50; color: white; padding: 10px 24px; border-radius: 8px; border: none; font-weight: bold; }
    .stButton>button:hover { background-color: #45a049; }
    .sidebar .sidebar-content { background-color: #f1f8e9; }
    h1, h2, h3 { color: #2e7d32; }
    .report-container { background-color: #f9f9f9; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50; }
    .healthy-tag { background-color: #4CAF50; color: white; padding: 5px 10px; border-radius: 4px; font-size: 14px; }
    .disease-tag { background-color: #f44336; color: white; padding: 5px 10px; border-radius: 4px; font-size: 14px; }
    .info-box { background-color: #e8f5e9; padding: 15px; border-radius: 8px; margin-top: 10px; }
    </style>
    """, unsafe_allow_html=True)

# -------------------------------------
# 📌 Backend API Configuration
# -------------------------------------
API_ENDPOINT = "http://localhost:8000"  # Change this to match your backend API endpoint

def analyze_image(image_file):
    """Send image to backend API for analysis"""
    try:
        # Convert image to base64
        img_bytes = image_file.getvalue()
        encoded = base64.b64encode(img_bytes).decode('utf-8')
        
        # Send to backend
        response = requests.post(
            f"{API_ENDPOINT}/predict",
            json={"image": encoded, "filename": image_file.name}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}", "details": response.text}
    except Exception as e:
        return {"error": f"Connection Error: {str(e)}"}

# -------------------------------------
# 📌 Initialize Session State
# -------------------------------------
if 'analyzed' not in st.session_state:
    st.session_state['analyzed'] = False
if 'prediction_result' not in st.session_state:
    st.session_state['prediction_result'] = None

# -------------------------------------
# 📌 Sidebar Navigation
# -------------------------------------
with st.sidebar:
    st.title("Plant Care")
    st.markdown("### 🌿 AI-Powered Plant Doctor")
    
    st.markdown("---")
    app_mode = st.radio("Navigation", ["Home", "Disease Recognition", "Plant Care Guide", "About"])
    
    st.markdown("---")
    st.markdown("#### How to use")
    st.info("""
    1. Upload a clear image of the plant leaf
    2. Click 'Analyze Image'
    3. Review the diagnosis and treatment plan
    """)
    
    st.markdown("---")
    st.markdown("#### Developed with ❤️ by Plant Care Team")

# -------------------------------------
# 📌 Main Page Logic
# -------------------------------------
if app_mode == "Home":
    st.title("🌿 Plant Disease Recognition System")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.image("homepage.jpg", use_column_width=True)
    
    with col2:
        st.markdown("""
        ## Welcome to Plant Care! 👋
        
        Our AI-powered tool helps you:
        
        - **Identify** plant diseases quickly and accurately
        - **Learn** about disease causes and symptoms
        - **Get** personalized treatment recommendations
        - **Prevent** future plant health issues
        
        Simply upload a photo of your plant's leaves, and our AI will do the rest!
        """)
    
    st.markdown("---")
    
    st.header("Featured Plant Diseases")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Tomato Late Blight")
        st.image("https://www.plantvillage.psu.edu/sites/plantvillage/files/Tomato_Late_blight1_1.jpg", use_column_width=True)
        st.markdown("A devastating disease causing brown lesions and white fuzzy growth.")
        
    with col2:
        st.subheader("Apple Scab")
        st.image("https://www.gardeningknowhow.com/wp-content/uploads/2020/11/apple-scab.jpg", use_column_width=True)
        st.markdown("Causes dark, scabby lesions on leaves and fruits.")
        
    with col3:
        st.subheader("Leaf Spot")
        st.image("https://www.planetnatural.com/wp-content/uploads/2012/12/leaf-spot.jpg", use_column_width=True)
        st.markdown("Common fungal infection with circular spots and yellowing.")

elif app_mode == "Disease Recognition":
    st.title("🔍 Disease Recognition")
    
    st.markdown("""
    Upload a clear image of your plant's leaf to diagnose any potential diseases.
    For best results, ensure good lighting and focus on the affected area.
    """)
    
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            analyze_clicked = st.button("🔬 Analyze Image", key="analyze_button")
            
            if analyze_clicked:
                with st.spinner('Analyzing your plant...'):
                    # Send to backend API
                    result = analyze_image(uploaded_file)
                    
                    if "error" in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        st.session_state['analyzed'] = True
                        st.session_state['prediction_result'] = result

        # Display results if analysis was successful
        if st.session_state['analyzed'] and st.session_state['prediction_result']:
            result = st.session_state['prediction_result']
            
            st.markdown("---")
            st.header("Diagnosis Results")
            
            # Result container
            st.markdown('<div class="report-container">', unsafe_allow_html=True)
            
            if "healthy" in result["disease_name"].lower():
                st.markdown(f'<span class="healthy-tag">HEALTHY</span>', unsafe_allow_html=True)
                st.success(f"Good news! Your plant appears to be healthy.")
            else:
                st.markdown(f'<span class="disease-tag">DISEASE DETECTED</span>', unsafe_allow_html=True)
                st.warning(f"Disease detected: {result['disease_name'].replace('___', ' - ').replace('_', ' ')}")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Don't show disease info for healthy plants
            if "healthy" not in result["disease_name"].lower():
                st.markdown("---")
                st.header("Disease Information")
                
                tabs = st.tabs(["Description", "Symptoms", "Treatment", "Prevention"])
                
                with tabs[0]:
                    st.markdown(result.get("description", "No description available"))
                
                with tabs[1]:
                    st.markdown(result.get("symptoms", "No symptoms information available"))
                
                with tabs[2]:
                    st.markdown(result.get("treatment", "No treatment information available"))
                    
                    st.markdown('<div class="info-box">', unsafe_allow_html=True)
                    st.markdown("⚠️ **Remember**: Always verify treatments with a professional before application.")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with tabs[3]:
                    st.markdown(result.get("prevention", "No prevention information available"))
                
                # Video suggestions
                if "videos" in result:
                    st.markdown("---")
                    st.subheader("Helpful Resources")
                    st.markdown(result["videos"])

elif app_mode == "Plant Care Guide":
    st.title("🌱 Plant Care Guide")
    
    st.markdown("""
    Explore our comprehensive guide to keeping your plants healthy and thriving.
    """)
    
    care_topics = [
        "Watering Basics", 
        "Nutrient Management", 
        "Pest Prevention", 
        "Seasonal Care",
        "Indoor Plant Tips"
    ]
    
    selected_topic = st.selectbox("Select a topic:", care_topics)
    
    if selected_topic == "Watering Basics":
        st.subheader("Watering Basics")
        st.markdown("""
        ### Key Principles
        - Water deeply but infrequently to encourage deep root growth
        - Water at the base of plants to avoid wet foliage
        - Morning watering is generally best
        
        ### Signs of Overwatering
        - Yellowing leaves
        - Soft, mushy stems
        - Mold or fungus on soil surface
        
        ### Signs of Underwatering
        - Wilting despite moist soil
        - Crispy, brown leaf edges
        - Slow growth
        """)
    
    elif selected_topic == "Nutrient Management":
        st.subheader("Nutrient Management")
        st.markdown("""
        ### Essential Nutrients
        - **Nitrogen (N)**: Leaf growth and green color
        - **Phosphorus (P)**: Root growth, flowering, fruiting
        - **Potassium (K)**: Overall plant health and disease resistance
        
        ### Organic vs. Synthetic Fertilizers
        Organic fertilizers release nutrients slowly and improve soil structure.
        Synthetic fertilizers provide immediate nutrients but don't improve soil.
        
        ### Application Tips
        - Follow package directions - more isn't better!
        - Apply fertilizers to moist soil to prevent root burn
        - Reduce fertilizer in fall/winter when growth slows
        """)

elif app_mode == "About":
    st.title("About Plant Care")
    
    st.markdown("""
    ## Our Mission
    
    At Plant Care, we believe everyone deserves healthy, thriving plants. Our AI-powered 
    tool makes professional plant disease diagnosis accessible to everyone, from hobby 
    gardeners to professional farmers.
    
    ## Technology
    
    Our system uses a deep learning convolutional neural network (CNN) trained on over 
    87,000 images of plant leaves across 38 different classes of plant diseases and healthy plants.
    
    ## Features
    
    - **Fast Analysis**: Get results in seconds
    - **Detailed Information**: Learn about causes, symptoms, and treatments
    - **Prevention Tips**: Avoid future outbreaks
    - **High Accuracy**: Our model achieves over 96% accuracy on test datasets
    
    ## Development Team
    
    Our interdisciplinary team brings together expertise in:
    - Machine Learning & AI
    - Plant Pathology
    - Agricultural Science
    - Software Development
    """)
    
    st.markdown("---")
    
    st.subheader("Privacy & Data Usage")
    st.info("""
    Your uploaded images are processed securely and are not stored permanently 
    unless you explicitly opt in to contribute to our dataset for improving the model.
    """)

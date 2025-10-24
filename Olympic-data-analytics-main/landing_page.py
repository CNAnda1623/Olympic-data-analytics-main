import streamlit as st
import base64
import os

def get_base64_image(image_path):
    """Convert local image to base64 for CSS background"""
    # Check if the file exists before trying to open it
    if not os.path.exists(image_path):
        st.error(f"Error: The background image was not found at '{image_path}'. Please check your file path.")
        return ""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return ""

def render_landing_page(img_base64):
    """Renders the fixed, non-scrollable landing page with the Olympic rings background."""
    
    # Conditional CSS injection (Anti-scrolling, Fixed Position, Olympic Background)
    # This overrides the global black background set in main.py
    st.markdown(f"""
    <style>
        /* Anti-scrolling rules */
        html, body {{
            overflow: hidden !important;
            height: 100vh !important;
        }}
        .stApp, [data-testid="stAppViewContainer"], .main {{
            overflow: hidden !important;
            height: 100vh !important;
        }}

        /* Apply Olympic Image Background only on the landing page, overriding the global black */
        /* Use raw img_base64 variable from main.py */
        .stApp {{
            background: linear-gradient(rgba(0, 0, 0, 0.3), rgba(0, 0, 0, 0.3)), 
                        url('data:image/jpeg;base64,{img_base64}');
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
            background-attachment: fixed;
        }}
        
        /* Lock the landing container to the viewport and center it horizontally */
        .landing-container {{
            position: fixed !important; 
            top: 0; 
            left: 50% !important;
            transform: translateX(-50%); 
            width: 90vw; 
            max-width: 1200px;
            height: 100vh; 
            z-index: 9999; 
            display: flex; 
            flex-direction: column;
            justify-content: space-between;
        }}
    </style>
    """, unsafe_allow_html=True)
    
    # HTML content for the landing page
    st.markdown("""
    <div class="landing-container">
        <h1 class="medal-title">MEDALMETRIX</h1>
        <div style="height: 1px;"></div> 
    </div>
    """, unsafe_allow_html=True)
    
    # Stop the script execution here to prevent loading the main app
    st.stop()

# End of landing_page.py
import streamlit as st
import traceback
import os
from pathlib import Path

# Import functions from modules
from analysis_view import safe_read_csv, safe_preprocess, render_analysis_view
from landing_page import render_landing_page, get_base64_image

# --- 1. CONFIGURATION AND GLOBAL CSS ---
st.set_page_config(page_title="MedalMetrix - Olympic Data Analytics", layout="wide", initial_sidebar_state="collapsed")

# Load image data for CSS injection with the CORRECT path for your structure
# Assumes you run 'streamlit run app/main.py' from the project root directory
CURRENT_DIR = Path(__file__).parent 
IMG_PATH = CURRENT_DIR.parent / "images" / "olymic_image.jpg"
img_base64 = get_base64_image(str(IMG_PATH))


# Global default CSS (Applies to ALL pages)
st.markdown(f"""
<style>
    /* GLOBAL DEFAULT: Solid Black background for the analysis page */
    .stApp {{
        background: black;
        background-attachment: fixed;
    }}

    /* GLOBAL STYLING: Landing container padding (Vertical Position Fix) */
    .landing-container {{
        display: flex;
        flex-direction: column;
        justify-content: space-between; 
        align-items: center;
        min-height: 100vh; 
        height: 100vh;
        text-align: center;
        /* Increased top padding to push the title down */
        padding: 15vh 20px 40px 20px; 
    }}
    
    /* GLOBAL STYLING: MedalMetrix title */
    .medal-title {{
        font-size: clamp(4rem, 10vw, 8rem); 
        font-weight: 900;
        letter-spacing: 0.1em;
        background: linear-gradient(90deg, #0085C7, #000000, #DF0024, #F4C300, #009F3D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 30px rgba(255, 255, 255, 0.3);
        margin-bottom: 0; 
        animation: glow 2s ease-in-out infinite alternate;
        margin-top: 0; 
    }}

    @keyframes glow {{
        from {{ filter: drop-shadow(0 0 10px rgba(255, 255, 255, 0.3)); }}
        to {{ filter: drop-shadow(0 0 20px rgba(255, 255, 255, 0.6)); }}
    }}
    
    /* GLOBAL STYLING: Overlay for charts */
    .app-overlay {{
        background: rgba(0, 0, 0, 0.75);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem;
        height: auto; 
        overflow: visible; 
    }}
    /* Removed Start Button CSS since it was deleted */
    
</style>
""", unsafe_allow_html=True)


# --- 2. FILE UPLOAD & LANDING PAGE LOGIC ---
st.sidebar.title("Olympic Data Upload")

# File uploaders MUST be in the main app to maintain session state
ath_file = st.sidebar.file_uploader("Upload athlete_events.csv", type=["csv"])
reg_file = st.sidebar.file_uploader("Upload noc_regions.csv", type=["csv"])

# Check if files are uploaded - this is the condition for the landing page
if ath_file is None or reg_file is None:
    # Render the landing page with the Olympic image background
    render_landing_page(img_base64) 
    # st.stop() is handled inside render_landing_page

# --- 3. DATA PROCESSING AND ANALYSIS VIEW ---
st.sidebar.title("Olympic Data Debug")
st.sidebar.markdown("Uploaded files will be processed below.")

try:
    # Safely read data from uploaded objects
    raw_ath_df = safe_read_csv(ath_file)
    raw_reg_df = safe_read_csv(reg_file)

    # Show debug info in sidebar
    with st.sidebar.expander("🔎 Uploaded files debug info"):
        st.write("athlete_events columns:", list(raw_ath_df.columns))
        st.write("noc_regions columns:", list(raw_reg_df.columns))
        st.write("athlete_events head:")
        st.dataframe(raw_ath_df.head(3))
        st.write("noc_regions head:")
        st.dataframe(raw_reg_df.head(3))

    # Preprocess the data
    df = safe_preprocess(raw_ath_df, raw_reg_df)

    # Render the analysis dashboard
    render_analysis_view(df)

except Exception:
    st.error("An error occurred during data processing. See sidebar for debug info.")
    st.text(traceback.format_exc())

# End of main.py
import streamlit as st
import json
import pandas as pd
import numpy as np
import cv2
from PIL import Image

# --- PAGE STYLING ---
# Custom CSS for a cleaner look and readable text
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    
    /* This targets the metric card */
    [data-testid="stMetric"] {
        background-color: #ffffff; 
        padding: 15px; 
        border-radius: 10px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }
    
    /* This forces the label and value to be dark gray/black */
    [data-testid="stMetricLabel"] p, [data-testid="stMetricValue"] div {
        color: #1f2937 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR & DATABASE ---
with st.sidebar:
    st.title("⚙️ Settings")
    try:
        with open('carbon_data.json', 'r') as f:
            carbon_db = json.load(f)
        st.success("Database Online")
    except:
        st.error("Database Offline")
    
    st.divider()
    st.info("Built with BART (Zero-Shot) & EasyOCR for Sustainable Shopping.")

# --- MAIN UI ---
st.title("🌱 EcoScan")
st.caption("Empowering your grocery choices with AI-driven carbon insights.")

if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False

if not st.session_state.models_loaded:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.info("Welcome! Please initialize the AI engines to start scanning.")
        if st.button("🚀 Wake Up AI Engines", use_container_width=True):
            with st.spinner("Loading AI Brains..."):
                import easyocr
                from transformers import pipeline
                st.session_state.classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-3")
                st.session_state.reader = easyocr.Reader(['en'])
                st.session_state.models_loaded = True
                st.rerun()

# --- APP ACTIVE STATE ---
if st.session_state.models_loaded:
    tab1, tab2 = st.tabs(["📤 Scan Receipt", "📈 Trends & Insights"])

    with tab1:
        col_up, col_img = st.columns([1, 1])
        
        with col_up:
            uploaded_file = st.file_uploader("Upload your receipt image here", type=["jpg", "jpeg", "png"])
            analyze_btn = st.button("🔍 Analyze Receipt", use_container_width=True, type="primary")

        if uploaded_file:
            raw_img = Image.open(uploaded_file)
            with col_img:
                st.image(raw_img, caption="Preview", width='stretch')

            if analyze_btn:
                with st.spinner("AI is reading your receipt..."):
                    # Process Image
                    img_array = np.array(raw_img)
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
                    processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                    
                    # OCR & AI Analysis
                    text_results = st.session_state.reader.readtext(processed, detail=0)
                    full_text = " ".join(text_results).lower()
                    labels = list(carbon_db.keys())
                    results = st.session_state.classifier(full_text, labels, multi_label=True)

                    # Store results in Session State to show in Tab 2
                    st.session_state.results_list = []
                    for label, score in zip(results['labels'], results['scores']):
                        if score > 0.25:
                            st.session_state.results_list.append({"Item": label.title(), "CO2": carbon_db[label]['co2'], "Swap": carbon_db[label]['swap']})

        if "results_list" in st.session_state:
            st.subheader("🛍️ Detected Items")
            for item in st.session_state.results_list:
                with st.expander(f"📌 {item['Item']}"):
                    c1, c2 = st.columns(2)
                    c1.metric("Carbon Cost", f"{item['CO2']} kg")
                    c2.info(f"💡 Suggestion: Try **{item['Swap']}** instead!")

    with tab2:
        if "results_list" in st.session_state and st.session_state.results_list:
            df = pd.DataFrame(st.session_state.results_list)
            
            # 1. Define total_carbon here so it is available for the score
            total_carbon = df["CO2"].sum()
            
            # 2. Metric Display
            st.metric("Total Trip Footprint", f"{total_carbon:.2f} kg CO2", delta_color="inverse")
            
            # 3. --- ECO-SCORE LOGIC ---
            # Now total_carbon is defined, so this won't crash!
            eco_score = max(0, 100 - int(total_carbon * 5))
            
            st.divider()
            c_score, c_label = st.columns([1, 2])
            
            with c_score:
                st.header(f"🏆 {eco_score}/100")
                st.caption("Your Sustainability Grade")
            
            with c_label:
                if eco_score > 80:
                    st.success("🌟 Earth Warrior! Your basket is highly sustainable.")
                elif eco_score > 50:
                    st.warning("⚖️ Balanced. Consider plant-based swaps next time.")
                else:
                    st.error("⚠️ High Impact. Check the 'Swaps' in Tab 1!")

            st.divider()
            st.subheader("📊 Visual Breakdown")
            st.bar_chart(df.set_index("Item")["CO2"])
            
        else:
            # High-engagement empty state
            st.info("🕒 Your insights are waiting!")
            st.write("Once you upload a receipt in the first tab, we'll calculate your score and impact here.")
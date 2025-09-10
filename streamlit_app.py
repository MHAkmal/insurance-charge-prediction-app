import streamlit as st
import polars as pl
import joblib
from streamlit_option_menu import option_menu
import pandas as pd  # Using pandas for easier display formatting in st.dataframe
import streamlit.components.v1 as components  # Added for HTML embedding
import base64  # To embed icons

# --- Page layout configuration ---
st.set_page_config(
    page_title="Insurance Charge Predictor",
    layout="wide"
)

# --- Custom CSS for styling ---
st.markdown(
    """
    <style>
    /* Sidebar width */
    [data-testid="stSidebar"] {
        width: 300px !important;
    }

    /* Container to help with alignment */
    .connect-box {
        display: flex;
        flex-direction: column;
        align-items: center; /* Center alignment */
        justify-content: center;
        height: 100%;
    }

    /* Style for connect links */
    .connect-container {
        display: flex;
        flex-direction: column;
        gap: 10px; /* Space between links */
    }
    .connect-link {
        display: flex;
        align-items: center;
        gap: 8px; /* Space between icon and text */
        text-decoration: none;
        font-weight: bold;
        color: #FFFFFF !important; /* White text for links */
    }
    .connect-link:hover {
        text-decoration: underline;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- SVGs for Icons (Base64 Encoded) ---
def get_svg_as_b64(svg_raw):
    """Encodes a raw SVG string into a Base64 string."""
    return base64.b64encode(svg_raw.encode('utf-8')).decode()

linkedin_svg = get_svg_as_b64('<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="#0077B5" stroke="currentColor" stroke-width="0" stroke-linecap="round" stroke-linejoin="round"><path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z"></path><rect x="2" y="9" width="4" height="12"></rect><circle cx="4" cy="4" r="2"></circle></svg>')
github_svg = get_svg_as_b64('<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="#FFFFFF" stroke="currentColor" stroke-width="0" stroke-linecap="round" stroke-linejoin="round"><path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path></svg>')
x_svg = get_svg_as_b64('<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 16 16" fill="#FFFFFF"><path d="M12.6.75h2.454l-5.36 6.142L16 15.25h-4.937l-3.867-5.07-4.425 5.07H.316l5.733-6.57L0 .75h5.063l3.495 4.633L12.602.75Zm-1.283 12.95h1.46l-7.48-10.74h-1.55l7.57 10.74Z"/></svg>')
threads_svg = get_svg_as_b64('<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22a10 10 0 1 1 0-20 10 10 0 0 1 0 20Z"></path><path d="M16.5 8.5c-.7-1-1.8-1.5-3-1.5s-2.3.5-3 1.5"></path><path d="M16.5 15.5c-.7 1-1.8 1.5-3 1.5s-2.3-.5-3-1.5"></path><path d="M8.5 12a5.5 5.5 0 1 0 7 0 5.5 5.5 0 0 0-7 0Z"></path></svg>')

# --- Caching the Model ---
@st.cache_resource
def load_model():
    """Loads the pre-trained insurance model pipeline."""
    try:
        model = joblib.load('insurance-charge-prediction-model.pkl')
        return model
    except FileNotFoundError:
        st.error("The model file 'insurance-charge-prediction-model.pkl' was not found. Please make sure the model is in the same directory as the script.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

# Load the model
model = load_model()

# --- Initialize session state for prediction history ---
if 'prediction_log' not in st.session_state:
    st.session_state.prediction_log = []

# --- Sidebar Navigation ---
with st.sidebar:
    selection = option_menu(
        menu_title="Main Menu",
        options=["Home", "Prediction", "Notebook"],
        icons=["house", "clipboard-data", "book"],
        menu_icon="cast",
        default_index=0,
    )

# --- Home Page ---
if selection == "Home":
    top_col1, top_col2 = st.columns([0.75, 0.25])

    with top_col1:
        st.title("🏥 Insurance Charge Prediction")
        st.markdown("""
        Welcome to the Insurance Charge Prediction app! This application predicts the estimated insurance charges for a potential customer based on their attributes.
        """)
        st.warning("Navigate to the **Prediction** tab from the sidebar to try the prediction yourself!", icon="👈")

    with top_col2:
        st.markdown('<div class="connect-box">', unsafe_allow_html=True)
        st.subheader("🔗 Connect With Me")
        linkedin_link = f'<a href="https://www.linkedin.com/in/mhakmal/" class="connect-link"><img src="data:image/svg+xml;base64,{linkedin_svg}" width="24"><span>MHAkmal</span></a>'
        github_link = f'<a href="https://github.com/MHAkmal" class="connect-link"><img src="data:image/svg+xml;base64,{github_svg}" width="24"><span>MHAkmal</span></a>'
        x_link = f'<a href="https://x.com/akmal621" class="connect-link"><img src="data:image/svg+xml;base64,{x_svg}" width="24"><span>MHAkmal</span></a>'
        threads_link = f'<a href="https://www.threads.com/@akmal621?__pwa=1" class="connect-link"><img src="data:image/svg+xml;base64,{threads_svg}" width="24"><span>MHAkmal</span></a>'
        st.markdown(f'<div class="connect-container">{linkedin_link}{github_link}{x_link}{threads_link}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    bp_col1, obj_col2 = st.columns(2)
    with bp_col1:
        st.header("Business Problem")
        st.write("How can insurance company provide accurate and instant insurance premium estimates to potential customers to increase conversion rates?")

    with obj_col2:
        st.header("Objective")
        st.write("Build a regression model to predict insurance charges based on customer attributes.")

    st.divider()

    st.header("Business Impact")
    st.subheader("Model Performance (Model: Linear Regression)")

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.metric(label="Mean Absolute Error (MAE)", value="$4,170")
    with m_col2:
        st.metric(label="Root Mean Squared Error (RMSE)", value="$6,037")

    st.subheader("Quote Accuracy Impact (per 100 quotes)")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Before Modeling")
        st.markdown("""
        - Manual or table-based quoting led to inaccuracies.
        - Inaccurate quotes could deter potential customers.
        """)
        st.metric(label="Quotes Converted to Policies (out of 100)", value="50")

    with col2:
        st.markdown("#### After Modeling")
        st.markdown("""
        - The model provides data-driven, personalized quotes instantly.
        - Assuming the accurate car price are those with error/diff
        - With a **76% accuracy rate**, customer trust and conversion increase.
        """)
        st.metric(label="Quotes Converted to Policies (out of 100)", value="76", delta="26 policies")

    st.success("Implementing this model could lead to an estimated **26 additional policies** for every 100 quotes provided.")

# --- Prediction Page ---
if selection == "Prediction" and model is not None:
    st.title("Predict Insurance Charges")
    st.warning("Adjust the parameters in the sections below, then scroll down and click the 'Predict Charges' button to see the result.", icon="ℹ️")
    st.header("Enter Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Categorical Input")
        sex = st.selectbox('Sex', ['female', 'male'], key='sex')
        smoker = st.selectbox('Smoker', ['no', 'yes'], key='smoker')
        region = st.selectbox('Region', ['southwest', 'southeast', 'northwest', 'northeast'], key='region')

    with col2:
        st.subheader("Numerical Input")
        age = st.slider('Age', min_value=18, max_value=64, value=25, key='age')
        bmi = st.slider('Body Mass Index (BMI)', min_value=15.0, max_value=55.0, value=25.0, step=0.1, format="%.1f", key='bmi')
        children = st.slider('Number of Children', min_value=0, max_value=5, value=0, key='children')


    if st.button('Predict Charges', type="primary"):
        current_input = {
            'age': age,
            'sex': sex,
            'bmi': bmi,
            'children': children,
            'smoker': smoker,
            'region': region
        }
        is_duplicate = any(log_entry['input'] == current_input for log_entry in st.session_state.prediction_log)

        if is_duplicate:
            st.warning("This exact prediction has already been made. Please see the history below.")
        else:
            input_data = pl.DataFrame({key: [value] for key, value in current_input.items()})
            prediction = model.predict(input_data)[0]
            st.subheader("Predicted Insurance Charge")
            st.success(f"**${prediction:,.2f}**")
            st.session_state.prediction_log.insert(0, {'input': current_input, 'prediction': prediction})

    st.divider()
    st.header("Prediction History")
    if st.session_state.prediction_log:
        history_df = pd.DataFrame([entry['input'] | {'Predicted Charge': entry['prediction']} for entry in st.session_state.prediction_log])
        st.dataframe(history_df.style.format({'Predicted Charge': '${:,.2f}', 'bmi': '{:.1f}'}), use_container_width=True)
    else:
        st.info("No predictions have been made in this session yet.")

elif selection == "Prediction" and model is None:
    st.warning("The prediction model is not available. Please check for error messages when the app started.")

# --- Notebook Page ---
if selection == "Notebook":
    st.title("Insurance Charge Prediction Model Notebook")
    st.write("This is a display of the Notebook used for data exploration, cleaning, and model building, converted to HTML.")
    st.info("The notebook below is a static HTML file.")

    notebook_filename = "insurance-charge-prediction_pandas.html"
    try:
        with open(notebook_filename, "r", encoding="utf-8") as f:
            html_data = f.read()
        components.html(html_data, width=None, height=800, scrolling=True)
    except FileNotFoundError:
        st.error(f"File not found: '{notebook_filename}'. Please ensure the notebook has been exported to HTML and is in the same directory as this script.")

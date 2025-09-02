import streamlit as st
import polars as pl
import joblib
from streamlit_option_menu import option_menu

st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        width: 250px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_model():
    """Loads the pre-trained insurance model pipeline."""
    # Ensure the model file path is correct
    model = joblib.load('insurance_charge_model.pkl')
    return model

# Load the model
model = load_model()

# --- Sidebar Navigation ---
with st.sidebar:
    selection = option_menu(
        menu_title="Main Menu",  # required
        options=["Home", "Prediction"],  # required
        icons=["house", "clipboard-data"],  # optional
        menu_icon="cast",  # optional
        default_index=0,  # optional
    )

# --- Home Page ---
if selection == "Home":
    st.title("🏥 Insurance Charge Prediction")
    st.markdown("""
    Welcome to the Insurance Charge Prediction app!

    This application predicts the estimated insurance charges for a potential customer based on their attributes.

    **How to use the app:**
    1.  Navigate to the **Prediction** tab from the sidebar menu.
    2.  Enter the customer's details using the interactive input fields.
    3.  Click on the **Predict Charges** button to see the estimated insurance cost.
    """)

# --- Prediction Page ---
if selection == "Prediction":
    st.title("Predict Insurance Charges")
    st.header("Enter Customer Details")

    # Input widgets for the 6 features
    age = st.slider('Age', min_value=18, max_value=64, value=25)
    bmi = st.number_input('Body Mass Index (BMI)', min_value=15.0, max_value=55.0, value=25.0, format="%.2f")
    children = st.slider('Number of Children', min_value=0, max_value=5, value=0)
    sex = st.selectbox('Sex', ['female', 'male'])
    smoker = st.selectbox('Smoker', ['no', 'yes'])
    region = st.selectbox('Region', ['southwest', 'southeast', 'northwest', 'northeast'])

    # --- Prediction Logic ---
    if st.button('Predict Charges'):
        input_data = pl.DataFrame({
            'age': [age],
            'bmi': [bmi],
            'children': [children],
            'sex': [sex],
            'smoker': [smoker],
            'region': [region]
        })

        prediction = model.predict(input_data)[0]

        # 3. Display the result
        st.subheader("Predicted Insurance Charge")

        st.write(f"**${prediction:,.2f}**")




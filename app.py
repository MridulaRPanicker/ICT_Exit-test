import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model and preprocessing objects
model = joblib.load('random_forest_model.joblib')
scaler = joblib.load('min_max_scaler.joblib')
label_encoder = joblib.load('label_encoder.joblib')
house_data_app = pd.read_csv('house_data_for_app.csv')

# --- Streamlit UI ----
st.set_page_config(page_title="Bengaluru House Price Predictor", layout="wide")
st.title('Bengaluru House Price Prediction App')
st.write('Enter property details to get an estimated price and explore market insights.')

# Sidebar for user inputs
st.sidebar.header('Property Details Input')

# Location dropdown
# Ensure 'Other' is an option if it was in the training data
location_options = sorted(house_data_app['location'].unique().tolist())
selected_location = st.sidebar.selectbox('Location', location_options)

# BHK slider
min_bhk = int(house_data_app['bhk'].min())
max_bhk = int(house_data_app['bhk'].max())
selected_bhk = st.sidebar.slider('Number of BHK', min_bhk, max_bhk, 2)

# Total Square Feet input
selected_sqft = st.sidebar.number_input('Total Square Feet (sqft)', min_value=1.0, value=1200.0, step=10.0)

# Bathrooms slider
min_bath = int(house_data_app['bath'].min())
max_bath = int(house_data_app['bath'].max())
selected_bath = st.sidebar.slider('Number of Bathrooms', min_bath, max_bath, 2)

# Balcony slider
min_balcony = int(house_data_app['balcony'].min())
max_balcony = int(house_data_app['balcony'].max())
selected_balcony = st.sidebar.slider('Number of Balconies', min_balcony, max_balcony, 1)

# Area Type radio buttons
area_type_options = house_data_app['area_type'].unique().tolist()
selected_area_type = st.sidebar.radio('Area Type', area_type_options)

# --- Prediction Logic ---
if st.sidebar.button('Predict Price'):
    # Basic input validation
    if selected_sqft <= 0 or selected_bath <= 0 or selected_bhk <= 0:
        st.error("Total Square Feet, Bathrooms, and BHK must be positive values.")
    else:
        # Create a DataFrame for the input, matching the training features
        input_data = pd.DataFrame(np.zeros((1, len(model.feature_names_in_))), columns=model.feature_names_in_)

        # Handle location encoding
        try:
            location_encoded_val = label_encoder.transform([selected_location])[0]
        except ValueError:
            # If location not in top 20, it would have been mapped to 'Other' during training
            # Find the 'Other' encoding from label_encoder.classes_
            if 'Other' in label_encoder.classes_:
                location_encoded_val = label_encoder.transform(['Other'])[0]
            else:
                st.error("Selected location not recognized and 'Other' category is missing from encoder. Cannot predict.")
                st.stop()
        input_data['location_encoded'] = location_encoded_val

        # Handle area_type one-hot encoding
        area_type_col_name = f'area_type_{selected_area_type}'
        if area_type_col_name in input_data.columns:
            input_data[area_type_col_name] = 1

        # Populate other numerical features
        input_data['bhk'] = selected_bhk
        input_data['balcony'] = selected_balcony

        # Scale 'total_sqft' and 'bath'
        # Create a dummy DataFrame with original names for scaling
        temp_df_for_scaling = pd.DataFrame([[selected_sqft, selected_bath]], columns=['total_sqft', 'bath'])
        scaled_values = scaler.transform(temp_df_for_scaling)
        input_data['total_sqft'] = scaled_values[0, 0]
        input_data['bath'] = scaled_values[0, 1]

        # Make prediction
        predicted_price = model.predict(input_data)[0]
        st.success(f'### Predicted Price: **{predicted_price:.2f} Lakhs**')

# --- Visualization --- #
st.header('Market Insights')
st.write(f'Top 5 most expensive locations for {selected_bhk} BHK properties:')

# Filter data for selected BHK and sort by price_per_sqft to find most expensive locations
filtered_bhk_data = house_data_app[house_data_app['bhk'] == selected_bhk]

if not filtered_bhk_data.empty:
    avg_price_per_sqft_by_location = filtered_bhk_data.groupby('location')['price_per_sqft'].mean().sort_values(ascending=False).head(5)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=avg_price_per_sqft_by_location.values, y=avg_price_per_sqft_by_location.index, ax=ax, palette='viridis')
    ax.set_title(f'Top 5 Most Expensive Locations for {selected_bhk} BHK (Avg. Price per Sqft)')
    ax.set_xlabel('Average Price per Square Feet')
    ax.set_ylabel('Location')
    st.pyplot(fig)
else:
    st.info(f"No data available for {selected_bhk} BHK properties to generate market insights.")

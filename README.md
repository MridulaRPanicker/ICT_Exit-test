# Use me for understanding the project and running the application


*   `ICT_Exit_Test.ipynb`: The main Jupyter notebook detailing the entire data science pipeline, from data loading and cleaning to model training and evaluation.
*   `app.py`: The Python script for the Streamlit web application.
*   `Bengaluru_House_Data.csv`: A preprocessed version of the dataset specifically prepared for the Streamlit application.
*   `linear_regression_model.joblib`: The trained Linear Regression model saved for use in the Streamlit app.
*   `random_forest_model.joblib`: The trained Random Forest Classifier model saved for reference (though the regression model is used in the app for prediction of house price).
*   `min_max_scaler.joblib`: The trained Min-Max Scaler object.
*   `label_encoder.joblib`: The trained Label Encoder object for location features.
*   `requirements.txt`: A list of all Python dependencies required to run the project.

*   1. `ICT_Exit_Test.ipynb` - is already have all the codes executed and all the plots rendered.

*   2. Running the Streamlit Web Application (`app.py`)

To run the interactive web application for house price prediction, follow these steps:

1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Navigate to the project directory** where `app.py`, `Bengaluru_House_Data.csv`, and all `joblib` model files are located.

3.  **Create and activate a virtual environment** (recommended):
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    This command will launch the application in your default web browser (usually at `http://localhost:8501`). You can then interact with the sidebar controls to get house price predictions and view market insights.

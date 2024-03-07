import joblib
from sklearn.preprocessing import RobustScaler
import category_encoders as ce
import pandas as pd

# Load the model, encoder, and scaler
gnb = joblib.load("gb_model.sav")
encoder = joblib.load("model_encoder.sav")
scaler = joblib.load("model_scaler.sav")

categorical = ['Gender', 'Smoking', 'Hx Smoking', 'Hx Radiothreapy', 'Thyroid Function', 'Physical Examination', 'Adenopathy', 'Pathology', 'Focality', 'Risk', 'T', 'N', 'M', 'Stage', 'Response']

def preprocess_data(data, encoder, scaler):
    data = encoder.transform(data)
    cols = data.columns
    data = scaler.transform(data)
    data = pd.DataFrame(data, columns=cols)
    return data

def predict(data):
    # Preprocess the input data
    data = preprocess_data(data, encoder, scaler)

    # Make sure the columns match the ones used during training
    cols = data.columns
    data = pd.DataFrame(data, columns=cols)

    # Use the preprocessed data for prediction
    return gnb.predict(data)
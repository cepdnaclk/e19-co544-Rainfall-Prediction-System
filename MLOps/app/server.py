'''from fastapi import FastAPI, HTTPException
import joblib
import numpy as np

app = FastAPI()

# Load pre-trained models
model_vavuniya = joblib.load('app/model_vavuniya.joblib')
model_anuradhapura = joblib.load('app/model_anuradhapura.joblib')
model_maha = joblib.load('app/model_maha.joblib')

@app.get('/')
def read_root():
    return {'message': 'Rainfall Prediction API'}

@app.post('/predict')
def predict(data: dict):
    """
    Predicts the rainfall for given features.

    Args:
        data (dict): A dictionary containing the features to predict.
        e.g. {
                "features": [1, 0.5, 0.2, ...]
             }

    Returns:
        dict: A dictionary containing the predicted rainfall for each location.
    """
    try:
        features = np.array(data['features']).reshape(1, -1)

        # Predict using the models
        prediction_vavuniya = model_vavuniya.predict(features)[0]
        prediction_anuradhapura = model_anuradhapura.predict(features)[0]
        prediction_maha = model_maha.predict(features)[0]

        return {
            'predicted_rainfall': {
                'Vavuniya': prediction_vavuniya,
                'Anuradhapura': prediction_anuradhapura,
                'Maha Illuppallama': prediction_maha
            }
        }
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing data for {e}")

# For serving the FastAPI app, create an app instance
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)'''

'''

from fastapi import FastAPI, HTTPException
import joblib
import numpy as np

app = FastAPI()

# Load pre-trained models
model_vavuniya = joblib.load('app/model_vavuniya.joblib')
model_anuradhapura = joblib.load('app/model_anuradhapura.joblib')
model_maha = joblib.load('app/model_maha.joblib')

@app.get('/')
def read_root():
    return {'message': 'Rainfall Prediction API'}

@app.post('/predict/vavuniya')
def predict_vavuniya(data: dict):
    """
    Predicts the rainfall for Vavuniya.

    Args:
        data (dict): A dictionary containing the features to predict for Vavuniya.
        e.g. {
                "features": [1, 0.5, 0.2, ...]
             }

    Returns:
        dict: A dictionary containing the predicted rainfall for Vavuniya.
    """
    try:
        features = np.array(data['features']).reshape(1, -1)
        prediction_vavuniya = model_vavuniya.predict(features)[0]

        return {
            'predicted_rainfall': {
                'Vavuniya': prediction_vavuniya
            }
        }
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing data for {e}")

@app.post('/predict/anuradhapura')
def predict_anuradhapura(data: dict):
    """
    Predicts the rainfall for Anuradhapura.

    Args:
        data (dict): A dictionary containing the features to predict for Anuradhapura.
        e.g. {
                "features": [1, 0.5, 0.2, ...]
             }

    Returns:
        dict: A dictionary containing the predicted rainfall for Anuradhapura.
    """
    try:
        features = np.array(data['features']).reshape(1, -1)
        prediction_anuradhapura = model_anuradhapura.predict(features)[0]

        return {
            'predicted_rainfall': {
                'Anuradhapura': prediction_anuradhapura
            }
        }
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing data for {e}")

@app.post('/predict/maha')
def predict_maha(data: dict):
    """
    Predicts the rainfall for Maha Illuppallama.

    Args:
        data (dict): A dictionary containing the features to predict for Maha Illuppallama.
        e.g. {
                "features": [1, 0.5, 0.2, ...]
             }

    Returns:
        dict: A dictionary containing the predicted rainfall for Maha Illuppallama.
    """
    try:
        features = np.array(data['features']).reshape(1, -1)
        prediction_maha = model_maha.predict(features)[0]

        return {
            'predicted_rainfall': {
                'Maha Illuppallama': prediction_maha
            }
        }
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing data for {e}")

# For serving the FastAPI app, create an app instance
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
'''

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import joblib
import numpy as np

app = FastAPI()

# Load pre-trained models
model_vavuniya = joblib.load('app/model_vavuniya.joblib')
model_anuradhapura = joblib.load('app/model_anuradhapura.joblib')
model_maha = joblib.load('app/model_maha.joblib')

@app.get('/')
def read_root():
    return {'message': 'Rainfall Prediction API'}

# Define the input data model for the prediction
class PredictData(BaseModel):
    date: str  # Expecting date in "YYYY-MM-DD" format
    Vavuniya_lag1: float
    Vavuniya_lag2: float
    Vavuniya_lag3: float
    Vavuniya_rolling_mean3: float
    Vavuniya_rolling_mean7: float
    Anuradhapura_lag1: float
    Anuradhapura_lag2: float
    Anuradhapura_lag3: float
    Anuradhapura_rolling_mean3: float
    Anuradhapura_rolling_mean7: float
    Maha_lag1: float
    Maha_lag2: float
    Maha_lag3: float
    Maha_rolling_mean3: float
    Maha_rolling_mean7: float

def extract_dayofyear_and_year(date_str):
    """Convert date string to day of the year and year."""
    date = datetime.strptime(date_str, "%Y-%m-%d")
    day_of_year = date.timetuple().tm_yday
    year = date.year
    return day_of_year, year

@app.post('/predict/vavuniya')
def predict_vavuniya(data: PredictData):
    """
    Predicts the rainfall for Vavuniya.

    Args:
        data (PredictData): Input data containing the date and other features.

    Returns:
        dict: A dictionary containing the predicted rainfall for Vavuniya.
    """
    try:
        day_of_year, year = extract_dayofyear_and_year(data.date)
        features = np.array([
            day_of_year,
            year,
            data.Vavuniya_lag1,
            data.Vavuniya_lag2,
            data.Vavuniya_lag3,
            data.Vavuniya_rolling_mean3,
            data.Vavuniya_rolling_mean7
        ]).reshape(1, -1)

        prediction_vavuniya = model_vavuniya.predict(features)[0]

        return {
            'predicted_rainfall': {
                'Vavuniya': prediction_vavuniya
            }
        }
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing data for {e}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")

@app.post('/predict/anuradhapura')
def predict_anuradhapura(data: PredictData):
    """
    Predicts the rainfall for Anuradhapura.

    Args:
        data (PredictData): Input data containing the date and other features.

    Returns:
        dict: A dictionary containing the predicted rainfall for Anuradhapura.
    """
    try:
        day_of_year, year = extract_dayofyear_and_year(data.date)
        features = np.array([
            day_of_year,
            year,
            data.Anuradhapura_lag1,
            data.Anuradhapura_lag2,
            data.Anuradhapura_lag3,
            data.Anuradhapura_rolling_mean3,
            data.Anuradhapura_rolling_mean7
        ]).reshape(1, -1)

        prediction_anuradhapura = model_anuradhapura.predict(features)[0]

        return {
            'predicted_rainfall': {
                'Anuradhapura': prediction_anuradhapura
            }
        }
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing data for {e}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")

@app.post('/predict/maha')
def predict_maha(data: PredictData):
    """
    Predicts the rainfall for Maha Illuppallama.

    Args:
        data (PredictData): Input data containing the date and other features.

    Returns:
        dict: A dictionary containing the predicted rainfall for Maha Illuppallama.
    """
    try:
        day_of_year, year = extract_dayofyear_and_year(data.date)
        features = np.array([
            day_of_year,
            year,
            data.Maha_lag1,
            data.Maha_lag2,
            data.Maha_lag3,
            data.Maha_rolling_mean3,
            data.Maha_rolling_mean7
        ]).reshape(1, -1)

        prediction_maha = model_maha.predict(features)[0]

        return {
            'predicted_rainfall': {
                'Maha Illuppallama': prediction_maha
            }
        }
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing data for {e}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")

# For serving the FastAPI app, create an app instance
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)

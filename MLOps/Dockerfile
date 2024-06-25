FROM python:3.10.12

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir -r /code/requirements.txt

COPY ./app /code/app

COPY model_vavuniya.joblib /code/app/
COPY model_anuradhapura.joblib /code/app/
COPY model_maha.joblib /code/app/

EXPOSE 8000

ENV NAME RainfallPrediction

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
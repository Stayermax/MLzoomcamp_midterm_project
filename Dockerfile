FROM python:3.10

WORKDIR /service_code

COPY  ./requirements.txt /service_code/requirements.txt

RUN pip install --upgrade pip
RUN pip install -r /service_code/requirements.txt

COPY ./data /service_code/data
COPY ./classifier_model.pkl /service_code/classifier_model.pkl
COPY ./prediction_service.py /service_code/prediction_service.py

CMD ["uvicorn", "prediction_service:app", "--host", "0.0.0.0", "--port", "8000","--workers","1"]

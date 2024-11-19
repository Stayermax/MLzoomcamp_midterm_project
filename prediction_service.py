import pandas as pd
import pickle as pkl
import xgboost as xgb
from copy import deepcopy
from fastapi import FastAPI



def load_model():
    with open(model_path, 'rb') as model_file:
        model, dv = pkl.load(model_file)
    return model, dv

root_path = "/credit_rating_serivce/api"
app = FastAPI(
    docs_url=root_path + "/docs",
    openapi_url=root_path + "/v1/openapi.json",
    openapi_tags=[])

model_path = "classifier_model.pkl"
processed_dataset_path = "data/prepared_dataset.csv"
text_to_int_map = {'Good': 0, 'Poor': 1, 'Standard': 2}
int_to_text_map = {0: 'Good', 1: 'Poor', 2: 'Standard'}

df = pd.read_csv(processed_dataset_path)
model, dv = load_model()

@app.get(root_path + '/random_sample')
def get_random_sample():
    """
    :return: random sample dictionary ready for prediction
    """
    row = df.sample(n=1).to_dict(orient='records')[0]
    print(row)
    credit_score = row['credit_score']
    del row['credit_score']
    return {"customer_data": row, "actual_label": credit_score}


@app.post(root_path + '/predict_dict')
def predict_dict(sample_dict: dict):
    dtest = xgb.DMatrix(
        dv.transform(sample_dict),
        feature_names=list(dv.get_feature_names_out())
    )
    prediction = model.predict(dtest)[0]
    return int_to_text_map[prediction]



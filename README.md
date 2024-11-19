# MLzoomcamp_midterm_project
ML zoomcamp midterm project

My dataset: [Credit score classification](https://www.kaggle.com/datasets/parisrohan/credit-score-classification)

To create environment

    conda create -n vitaly_venv python=3.10

To create environment with needed dependencies 

    conda env create -f environment.yml



In order to run prediction service:

    uvicorn prediction_service:app --reload --port 8000 --host localhost

In order to build and run prediction service in Docker container:
    
    docker build . -t prediction_service
    docker run -d -p 8000:8000 prediction_service
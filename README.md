# Heart Disease Prediction Pipeline

This repository contains a prototype machine learning pipeline for predicting whether a patient will develop heart disease.

### Features
* End-to-end pipeline: data ingestion to model prediction.
* Data preprocessing: gandles missing values etc.
* Model training: random-forest for classification.
* Streamlit app: A web interface for interacting with the prediction model.
* Docker integration: For easy setup and deployment.

### Installation
Clone the repository:
```commandline
git clone https://github.com/tobyhowison/mf-prototype-pipeline.git
cd mf-prototype-pipeline
```
Install dependencies:
```commandline
pip install -r requirements.txt
```

### Training
Update path to training data in config_params.py, e.g.:
```commandline
DATA_PATH = 'data/patient_heart_data.csv'
```
Run the script to train the model:
```commandline
python pipeline/train_model.py
```
This will preprocess the data and train the classification model. The trained model will be saved in the model/ directory for future use.

### Use the trained model

#### Streamlit app
Run the Streamlit app to make new evaluations on the model:
```commandline
streamlit run app.py
```
Navigate to the app page in your browser at:
http://localhost:8501

#### Docker
You can run the streamlit app using Docker. Build the Docker image:
```commandline
docker build -t heart-disease-predictor .
```
Run the container:
```commandline
docker run -p 8080:8080 heart-disease-predictor
```
Access model at `http://0.0.0.0:8080` or `http://localhost:8080`

### Deploy to google cloud
The docker container can be deployed on google cloud, allowing remote access to the streamlit app.

```commandline
docker build -t gcr.io/[YOUR GCP PROJECT ID]/heart-prediction-model . 
```

```commandline
docker push gcr.io/[YOUR GCP PROJECT ID]/heart-prediction-model  
```

```commandline
gcloud run deploy --image gcr.io/[YOUR GCP PROJECT ID]/heart-prediction-model --platform managed
```

Access deployed model, e.g.
https://heart-prediction-model-796115071536.europe-north1.run.app

### Project Structure

* `app.py`: Main application file (Streamlit app).
* `config_params.py`: Configuration parameters for the pipeline.
* `model/: Model saves here
* `pipeline/`: Pipeline classes.
* `data/`: Dataset storage.

### Questions for the client
* What does -99.99 mean in oldpeak
* Realistic ranges for various features

### Future improvements
* Documentation: improve documentation.
* CI/CD Integration: Implement continuous integration (CI) with automated testing and deployment (e.g., GitHub Actions).
* Cloud Deployment: Add a pipeline for deploying to GCP or similar.
* Model Testing: Introduce unit and integration tests for the pipeline.
* Parameter Tuning: Include hyperparameter tuning and model versioning.
* API Integration: Provide an API for making predictions.
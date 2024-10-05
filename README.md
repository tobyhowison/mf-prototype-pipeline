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

### Training: 
Prepare the dataset: Ensure that the training data (`patient_heart_data.csv`) is placed in the data/ folder.

Run the script to train the model:
```commandline
python model/train_model.py
```
This will preprocess the data and train the classification model.
Save the model: The trained model will be saved in the model/ directory for future use.

### Use the Streamlit App
Run the Streamlit app to make new evaluations on the model:
```commandline
streamlit run app.py
```

Access the app: Navigate to the app page in your browser at:
http://localhost:8501

### Docker
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
The docker container can be deployed on google cloud (https://cloud.google.com/?hl=en), allowing remote access to the streamlit app.

```commandline
docker build -t gcr.io/[YOUR GCP PROJECT ID]/heart-prediction-model . 
```

```commandline
docker push gcr.io/[YOUR GCP PROJECT ID]/heart-prediction-model  
```

```commandline
gcloud run deploy --image gcr.io/[YOUR GCP PROJECT ID]/heart-prediction-model --platform managed
```

Access deployed model, e.g.:

https://heart-prediction-model-796115071536.europe-north1.run.app



### Project Structure

* `app.py`: Main application file (Streamlit app).
* `config_params.py`: Configuration parameters for the pipeline.
* `model/: Saved models.
* `pipeline/`: Machine learning pipeline.
* `data/`: Dataset storage.
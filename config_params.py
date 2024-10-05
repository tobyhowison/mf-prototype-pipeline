DATA_PATH = 'data/patient_heart_data.csv'
MODEL_SAVE_PATH = 'model/model.pkl'
RANDOM_SEED = 102938475
CATEGORICAL_FEATURES: list[str] = [
    'age',
    'sex',
    'chest pain type',
    'fasting blood sugar',
    'resting ECG',
    'exang',
    'slope',
    'number vessels flourosopy',
    'thal'
]
CONTINUOUS_FEATURES: list[str] = [
    'resting blood pressure',
    'chol',
    'max heart rate',
    'oldpeak']
CATEGORICAL_IMPUTATION_STRATEGY: str = 'most_frequent'
CONTINUOUS_IMPUTATION_STRATEGY: str = 'mean'
CONTINUOUS_REALISTIC_LIMITS: dict[str, tuple[int, int]] = {
    'age': (0, 125),
    'resting blood pressure': (0, 250),
    'chol': (0, 600),
    'max_heart_rate': (50, 250),
    'oldpeak': (0, 10),
}

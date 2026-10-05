FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY model.joblib .

# Load the trained model and run a sample prediction on one Breast Cancer record
CMD ["python", "-c", "import joblib; from sklearn.datasets import load_breast_cancer; m = joblib.load('model.joblib'); d = load_breast_cancer(); p = m.predict(d.data[:1])[0]; print('Prediction:', d.target_names[p])"]

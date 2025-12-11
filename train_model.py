# backend/train_model.py
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
from data_generator import generate_sample_events
df = generate_sample_events(1500)
df['start_ts'] = pd.to_datetime(df['start_ts'])
df['hour'] = df['start_ts'].dt.hour
X = pd.concat([df[['duration_mins','hour']], pd.get_dummies(df['department'], prefix='dept')], axis=1)
model = IsolationForest(n_estimators=100, contamination=0.08, random_state=42)
model.fit(X)
joblib.dump({"model": model, "columns": X.columns.tolist()}, "model.pkl")
print("Saved model.pkl")

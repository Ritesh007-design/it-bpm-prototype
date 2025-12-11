# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd, joblib
from data_generator import generate_sample_events
from sklearn.ensemble import IsolationForest
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
try:
    info = joblib.load("model.pkl")
    model, cols = info["model"], info["columns"]
except:
    df = generate_sample_events(1500)
    df['start_ts']=pd.to_datetime(df['start_ts'])
    df['hour']=df['start_ts'].dt.hour
    X=pd.concat([df[['duration_mins','hour']],pd.get_dummies(df['department'],prefix='dept')],axis=1)
    model=IsolationForest(contamination=0.08).fit(X)
    joblib.dump({"model":model,"columns":X.columns.tolist()},"model.pkl")
    cols=X.columns
DF = generate_sample_events(600)
DF['start_ts']=pd.to_datetime(DF['start_ts'])
def prep(df):
    df2=df.copy()
    df2['hour']=df2['start_ts'].dt.hour
    X=pd.concat([df2[['duration_mins','hour']],pd.get_dummies(df2['department'],prefix='dept')],axis=1)
    for c in cols:
        if c not in X: X[c]=0
    return X[cols]
@app.get("/health")
def health(): return {"status":"ok"}
@app.get("/metrics")
def metrics():
    d=DF.copy()
    return {
        "total_tasks":len(d),
        "completed":int((d['status']=="COMPLETED").sum()),
        "delayed":int((d['duration_mins']>60).sum()),
        "failed":int((d['status']=="FAILED").sum()),
        "avg_duration":float(d['duration_mins'].mean())
    }
@app.get("/predict")
def predict(limit:int=50):
    d=DF.tail(limit).copy()
    X=prep(d)
    scores=model.decision_function(X)
    preds=model.predict(X)
    d['anomaly']=preds==-1
    d['anomaly_score']=-scores
    return d.to_dict(orient="records")

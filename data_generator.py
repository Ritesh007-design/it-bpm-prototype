# backend/data_generator.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
def generate_sample_events(n=1000, seed=42):
    np.random.seed(seed)
    base = datetime.now() - timedelta(days=30)
    rows = []
    for i in range(n):
        task_id = f"TASK-{1000+i}"
        duration = max(1, int(np.random.normal(30,20)))
        if np.random.rand() < 0.08:
            duration = int(np.random.normal(240,60))
        start = base + timedelta(minutes=int(np.random.rand()*60*24*30))
        end = start + timedelta(minutes=duration)
        status = np.random.choice(["COMPLETED","PENDING","FAILED"], p=[0.8,0.15,0.05])
        rows.append({
            "task_id": task_id,
            "start_ts": start,
            "end_ts": end,
            "duration_mins": duration,
            "status": status,
            "owner": f"owner_{np.random.randint(1,8)}",
            "department": np.random.choice(["Ops","Finance","HR","Support"])
        })
    return pd.DataFrame(rows)
if __name__ == "__main__":
    df = generate_sample_events()
    df.to_csv("sample_events.csv", index=False)
    print("Saved sample_events.csv")

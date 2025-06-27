from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
import pandas as pd
from prophet import Prophet

app = FastAPI()

class ForecastRequest(BaseModel):
    dates: List[str]
    values: List[float]
    horizon: int

@app.post("/forecast")
async def forecast(data: ForecastRequest):
    df = pd.DataFrame({"ds": data.dates, "y": data.values})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=data.horizon)
    forecast = model.predict(future)
    return {
        "dates": forecast["ds"].astype(str).tolist(),
        "forecast": forecast["yhat"].round(2).tolist(),
        "upper": forecast["yhat_upper"].round(2).tolist(),
        "lower": forecast["yhat_lower"].round(2).tolist()
    }

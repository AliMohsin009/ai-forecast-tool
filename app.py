from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from prophet import Prophet
from neuralprophet import NeuralProphet
import uvicorn
import os

app = FastAPI()

# === CORS Setup (Allow all for dev — restrict in production) ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Request Schema ===
class ForecastRequest(BaseModel):
    dates: List[str]
    values: List[float]
    regressors: Dict[str, List[float]] = {}
    horizon: int
    scenario: float = 0.0
    model_type: Optional[str] = "prophet"  # prophet, neuralprophet, auto
    hyperparameters: Optional[Dict[str, float]] = {}

# === Forecast Endpoint ===
@app.post("/forecast")
async def forecast(data: ForecastRequest):
    try:
        if len(data.dates) != len(data.values):
            raise HTTPException(status_code=400, detail="Dates and values must have the same length.")

        df = pd.DataFrame({
            "ds": pd.to_datetime(data.dates),
            "y": data.values
        })

        # Add regressors to dataframe
        for name, values in data.regressors.items():
            if len(values) != len(data.dates):
                raise HTTPException(
                    status_code=400,
                    detail=f"Regressor '{name}' has length {len(values)} but expected {len(data.dates)}."
                )
            df[name] = values

        def extend_regressors(original, horizon, scenario):
            last_val = original.iloc[-1]
            return original.tolist() + [last_val * (1 + scenario / 100)] * horizon

        # === Prophet Runner ===
        def run_prophet():
            model = Prophet(
                seasonality_mode=data.hyperparameters.get("seasonality_mode", "additive")
            )
            for name in data.regressors:
                model.add_regressor(name)
            model.fit(df)
            future = model.make_future_dataframe(periods=data.horizon)
            for name in data.regressors:
                future[name] = extend_regressors(df[name], data.horizon, data.scenario)
            forecast = model.predict(future)
            return forecast, model.predict(df)["yhat"], model

        # === NeuralProphet Runner ===
        def run_neuralprophet():
            model = NeuralProphet()
            for name in data.regressors:
                model.add_future_regressor(name)
            model.fit(df, freq="D")
            future = model.make_future_dataframe(df, periods=data.horizon, n_historic_predictions=False)
            for name in data.regressors:
                future[name] = extend_regressors(df[name], data.horizon, data.scenario)
            forecast = model.predict(future)
            return forecast, model.predict(df)["yhat1"], model

        results = []

        # === Model Selection ===
        if data.model_type == "prophet":
            forecast_df, train_pred, _ = run_prophet()
            results.append(("prophet", forecast_df, train_pred))
        elif data.model_type == "neuralprophet":
            forecast_df, train_pred, _ = run_neuralprophet()
            results.append(("neuralprophet", forecast_df, train_pred))
        elif data.model_type == "auto":
            p_forecast, p_pred, _ = run_prophet()
            n_forecast, n_pred, _ = run_neuralprophet()
            p_mape = mean_absolute_percentage_error(df["y"], p_pred)
            n_mape = mean_absolute_percentage_error(df["y"], n_pred)
            if p_mape <= n_mape:
                results.append(("prophet", p_forecast, p_pred))
            else:
                results.append(("neuralprophet", n_forecast, n_pred))
        else:
            raise HTTPException(status_code=400, detail="Invalid model_type provided.")

        # === Output formatting ===
        model_name, forecast_df, train_pred = results[0]
        forecast_col = "yhat1" if model_name == "neuralprophet" else "yhat"
        result = forecast_df.tail(data.horizon)

        mape = float(mean_absolute_percentage_error(df["y"], train_pred)) * 100
        rmse = float(np.sqrt(mean_squared_error(df["y"], train_pred)))

        return {
            "model": model_name,
            "dates": result["ds"].astype(str).tolist(),
            "forecast": result[forecast_col].round(2).tolist(),
            "confidence_lower": result.get("yhat_lower", []).round(2).tolist() if "yhat_lower" in result else [],
            "confidence_upper": result.get("yhat_upper", []).round(2).tolist() if "yhat_upper" in result else [],
            "metrics": {
                "MAPE (%)": round(mape, 2),
                "RMSE": round(rmse, 2)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecasting error: {str(e)}")

# === Run on Render-compatible port ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Required for Render
    uvicorn.run(app, host="0.0.0.0", port=port)

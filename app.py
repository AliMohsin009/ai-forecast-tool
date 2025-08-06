from fastapi import FastAPI, Request, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Optional, Union
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import os
import stripe
import secrets
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error
from prophet import Prophet
from neuralprophet import NeuralProphet
from math import sqrt
import logging
import time

from supabase import create_client, Client

# === Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("forecast-api")

# === App Initialization ===
app = FastAPI()

# === Supabase Configuration ===
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://bhpqdxwveaafuztflewt.supabase.co")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "your_service_role_key_here")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# === Stripe Configuration ===
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
endpoint_secret = os.getenv("STRIPE_WEBHOOK_SECRET")

# === API Key Validation (Fixed) ===
def verify_api_key(request: Request):
    if request.url.scheme != "https":
        raise HTTPException(status_code=403, detail="HTTPS is required.")

    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing API key.")
    key = auth.split(" ")[1]

    response = supabase.table("api_keys").select("*").eq("key", key).execute()

    if response.status_code != 200 or not response.data:
        raise HTTPException(status_code=403, detail="Invalid or expired API key.")

    record = response.data[0]

    if "expires_at" in record:
        expires_str = record["expires_at"].replace("Z", "")
        expires_dt = datetime.fromisoformat(expires_str)
        if expires_dt < datetime.utcnow():
            raise HTTPException(status_code=403, detail="API key expired.")

    return record

# === Forecast Request Schema ===
class ForecastRequest(BaseModel):
    dates: List[str]
    values: List[float]
    regressors: Optional[Dict[str, List[float]]] = None
    horizon: int
    scenario: int
    model_type: str
    changepoint_prior_scale: Optional[float] = 0.05
    n_epochs: Optional[int] = 100
    hyperparams: Optional[Dict[str, Union[float, str, int]]] = None

# === Forecast Logic ===
def run_forecast_logic(req: ForecastRequest, model_name: str):
    df = pd.DataFrame({"ds": pd.to_datetime(req.dates).tz_localize(None), "y": req.values})
    if req.regressors:
        for k, v in req.regressors.items():
            df[k] = v

    if model_name == "prophet":
        cps = req.hyperparams.get("changepoint_prior_scale", 0.05)
        mode = req.hyperparams.get("seasonality_mode", "additive")
        model = Prophet(changepoint_prior_scale=cps, seasonality_mode=mode)
        if req.regressors:
            for col in req.regressors:
                model.add_regressor(col)
        model.fit(df)
        future = model.make_future_dataframe(periods=req.horizon)
        if req.regressors:
            for col in req.regressors:
                future[col] = df[col].tolist() + [df[col].iloc[-1]] * req.horizon
        forecast = model.predict(future)
        yhat = forecast["yhat"].tolist()
        lower = forecast["yhat_lower"].tolist()
        upper = forecast["yhat_upper"].tolist()
        ds = forecast["ds"].astype(str).tolist()

    elif model_name == "neuralprophet":
        n_epochs = req.hyperparams.get("n_epochs", 100)
        model = NeuralProphet(epochs=n_epochs)
        if req.regressors:
            for col in req.regressors:
                model.add_future_regressor(col)
        model.fit(df, freq="D")
        future = model.make_future_dataframe(df, periods=req.horizon)
        forecast = model.predict(future)
        yhat = forecast["yhat1"].tolist()
        lower = [val * 0.9 for val in yhat]
        upper = [val * 1.1 for val in yhat]
        ds = forecast["ds"].astype(str).tolist()
    else:
        raise ValueError("Unsupported model type.")

    rmse = sqrt(mean_squared_error(df["y"], yhat[:len(df)]))
    mape = np.mean(np.abs((df["y"] - yhat[:len(df)]) / df["y"])) * 100

    return {
        "model": model_name,
        "dates": ds,
        "forecast": np.round(yhat, 2).tolist(),
        "lower": np.round(lower, 2).tolist(),
        "upper": np.round(upper, 2).tolist(),
        "metrics": {
            "RMSE": round(rmse, 2),
            "MAPE (%)": round(mape, 2)
        }
    }

# === Forecast Endpoint ===
@app.post("/forecast")
async def forecast(data: ForecastRequest, user: dict = Depends(verify_api_key)):
    start_time = time.time()
    try:
        if len(data.dates) != len(data.values):
            raise HTTPException(status_code=400, detail="Dates and values must match in length.")

        model_types = ["prophet", "neuralprophet"] if data.model_type == "auto" else [data.model_type.lower()]
        prophet_cps = [0.05]
        seasonality_modes = ["additive"]
        neural_epochs = [100]

        best_result = None
        best_rmse = float("inf")
        best_meta = {}

        for model in model_types:
            if model == "prophet":
                for cps in prophet_cps:
                    for seasonality in seasonality_modes:
                        try:
                            result = run_forecast_logic(ForecastRequest(**data.dict(), model_type="prophet", hyperparams={"changepoint_prior_scale": cps, "seasonality_mode": seasonality}), model)
                            if result["metrics"]["RMSE"] < best_rmse:
                                best_result = result
                                best_rmse = result["metrics"]["RMSE"]
                                best_meta = {"model": "prophet", "changepoint_prior_scale": cps, "seasonality_mode": seasonality}
                        except Exception as e:
                            logger.warning(f"Prophet failed with cps={cps}, seasonality={seasonality}: {e}")
            elif model == "neuralprophet":
                for epochs in neural_epochs:
                    try:
                        result = run_forecast_logic(ForecastRequest(**data.dict(), model_type="neuralprophet", hyperparams={"n_epochs": epochs}), model)
                        if result["metrics"]["RMSE"] < best_rmse:
                            best_result = result
                            best_rmse = result["metrics"]["RMSE"]
                            best_meta = {"model": "neuralprophet", "n_epochs": epochs}
                    except Exception as e:
                        logger.warning(f"NeuralProphet failed with epochs={epochs}: {e}")

        if not best_result:
            raise HTTPException(status_code=500, detail="AutoML failed to produce a valid result.")

        best_result["auto_ml_config"] = best_meta
        logger.info(f"Forecast completed in {time.time() - start_time:.2f} seconds")
        return best_result

    except Exception as e:
        logger.error(f"Forecasting error: {e}")
        raise HTTPException(status_code=500, detail=f"Forecasting error: {str(e)}")

# === Stripe Webhook Endpoint ===
@app.post("/webhook/stripe")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get('stripe-signature')
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, endpoint_secret)
    except Exception as e:
        logger.warning(f"Stripe signature error: {e}")
        raise HTTPException(status_code=400, detail="Invalid signature")

    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        email = session.get("customer_email")
        plan = session.get("metadata", {}).get("plan", "basic")
        new_key = secrets.token_urlsafe(32)
        expires = datetime.utcnow() + timedelta(days=30)

        supabase.table("api_keys").insert([{
            "key": new_key,
            "user_id": email,
            "plan": plan,
            "expires_at": expires.isoformat()
        }]).execute()

        logger.info(f"[Stripe] Issued key for {email} plan={plan}")
    return {"status": "ok"}

# === Admin: Issue Key ===
@app.post("/admin/issue-key")
async def issue_api_key(user_email: str, plan: str, days_valid: int = 30):
    new_key = secrets.token_urlsafe(32)
    expires = datetime.utcnow() + timedelta(days=days_valid)

    supabase.table("api_keys").insert([{
        "key": new_key,
        "user_id": user_email,
        "plan": plan,
        "expires_at": expires.isoformat()
    }]).execute()

    logger.info(f"[Admin] Issued key for {user_email} plan={plan}")
    return {"api_key": new_key, "expires_at": expires}

# === Health Check ===
@app.get("/health")
async def health():
    return {"status": "ok"}

# === Run Locally or on Render ===
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port)

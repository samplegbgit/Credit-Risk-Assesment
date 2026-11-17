import traceback
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent

FEATURE_COLUMNS = [
    "person_age", "person_income", "person_home_ownership", "person_emp_length",
    "loan_intent", "loan_grade", "loan_amnt", "loan_int_rate",
    "loan_percent_income", "cb_person_default_on_file", "cb_person_cred_hist_length",
]

ml_model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_model["model"] = joblib.load(BASE_DIR / "credit_risk_model.pkl")
    # numpy.float32 can't be serialized to JSON -> convert to plain float
    ml_model["threshold"] = float(joblib.load(BASE_DIR / "best_threshold.pkl"))
    yield
    ml_model.clear()


app = FastAPI(lifespan=lifespan)


class LoanApplication(BaseModel):
    person_age: int = Field(ge=18, le=100)
    person_income: float = Field(gt=0)
    person_home_ownership: str
    person_emp_length: float = Field(ge=0, le=60)
    loan_intent: str
    loan_grade: str
    loan_amnt: float = Field(gt=0)
    loan_int_rate: float = Field(ge=0, le=40)
    loan_percent_income: float = Field(ge=0, le=1)
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int = Field(ge=0, le=60)


@app.post("/predict")
def predict(data: LoanApplication):
    try:
        record = data.model_dump() if hasattr(data, "model_dump") else data.dict()
        input_df = pd.DataFrame([record])[FEATURE_COLUMNS]

        probability = float(ml_model["model"].predict_proba(input_df)[:, 1][0])
        threshold = ml_model["threshold"]
        prediction = int(probability >= threshold)

        return {
            "default_probability": probability,
            "default_prediction": prediction,
            "threshold": threshold,
            "Result": "High Risk" if prediction == 1 else "Low Risk",
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")



app.mount("/", StaticFiles(directory=BASE_DIR / "static", html=True), name="static")
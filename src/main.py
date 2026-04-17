from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.schemas import SalesPredictionRequest, SalesPredictionResponse
from src.preprocessing import DataPreprocessor
from src.inference import ModelPredictor

app = FastAPI(
    title="Terapi Yazılım ERP API",
    description="Sade ve Güçlü Tahmin Modülü",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = ModelPredictor()
preprocessor = DataPreprocessor(model_instance=predictor.model)

@app.get("/health")
def health_check():
    return {"status": "ok", "model_ready": predictor.model is not None}

@app.post("/predict", response_model=SalesPredictionResponse)
def get_prediction(request: SalesPredictionRequest):
    try:
        raw_data = request.dict()
        df_processed = preprocessor.prepare_for_inference(raw_data)
        result = predictor.predict(df_processed)
        
        return SalesPredictionResponse(
            status="success",
            predicted_sales=result
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
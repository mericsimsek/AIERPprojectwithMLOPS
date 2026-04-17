from pydantic import BaseModel, Field
from typing import Optional

class SalesPredictionRequest(BaseModel):
    """API'ye gelecek isteğin anayasasıdır."""
    store: int = Field(..., description="Mağaza ID")
    item: int = Field(..., description="Ürün ID")
    year: int = Field(..., description="Yıl")
    month: int = Field(..., description="Ay")
    
    # Yeni eklediğimiz özelliklerin Python tarafında karşılanması
    day_of_week: Optional[int] = 0
    is_wknd: Optional[int] = 0
    holiday_effect: Optional[str] = "none"
    pricing_strategy: Optional[str] = "normal"
    weather_impact: Optional[str] = "normal"
    sales_trend_1month: Optional[float] = 1.05
    store_general_crowd: Optional[float] = 45.0
    
    class Config:
        # EĞER arayüzden burada yazmayan başka bir veri gelirse, çökertme, esnek ol!
        extra = "allow" 

class SalesPredictionResponse(BaseModel):
    status: str
    predicted_sales: int
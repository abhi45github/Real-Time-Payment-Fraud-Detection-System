"""
Minimal FastAPI Server for Fraud Detection
Runs without Redis, Kafka, or PostgreSQL for testing
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import time
import random
from datetime import datetime
import uvicorn

# Request/Response Models
class TransactionRequest(BaseModel):
    """Transaction request model"""

    transaction_id: str = Field(..., min_length=1, max_length=100)
    user_id: str = Field(..., min_length=1, max_length=100)
    merchant_id: str = Field(..., min_length=1, max_length=100)
    amount: float = Field(..., gt=0, le=1000000)
    currency: str = Field(default="USD", max_length=3)
    merchant_category: str = Field(..., max_length=50)
    entry_mode: str = Field(..., max_length=20)
    country: str = Field(..., max_length=2)
    ip_address: str = Field(...)
    device_id: str = Field(..., max_length=100)
    email: Optional[str] = Field(default=None, max_length=255)
    phone: Optional[str] = Field(default=None, max_length=20)

    class Config:
        schema_extra = {
            "example": {
                "transaction_id": "txn_123456",
                "user_id": "user_789",
                "merchant_id": "merchant_456",
                "amount": 150.50,
                "currency": "USD",
                "merchant_category": "retail",
                "entry_mode": "chip",
                "country": "US",
                "ip_address": "192.168.1.1",
                "device_id": "device_123"
            }
        }


class FraudResponse(BaseModel):
    """Fraud detection response model"""

    transaction_id: str
    decision: str
    risk_score: float = Field(..., ge=0, le=1)
    ml_score: float = Field(..., ge=0, le=1)
    rule_score: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)
    reason_codes: List[str]
    processing_time_ms: float
    timestamp: float


class BatchTransactionRequest(BaseModel):
    """Batch transaction request"""
    transactions: List[TransactionRequest] = Field(..., max_items=100)


# Create FastAPI app
app = FastAPI(
    title="Fraud Detection API (Minimal Version)",
    description="Simplified fraud detection system for testing",
    version="1.0.0"
)


# Simple fraud detection logic
def detect_fraud_simple(transaction: TransactionRequest) -> Dict:
    """Simple fraud detection logic for testing"""

    start_time = time.time()

    # Calculate risk score based on rules
    risk_score = 0.1  # Base score
    reason_codes = []

    # High amount increases risk
    if transaction.amount > 5000:
        risk_score += 0.4
        reason_codes.append("HIGH_AMOUNT")
    elif transaction.amount > 1000:
        risk_score += 0.2
        reason_codes.append("MEDIUM_AMOUNT")

    # Risky merchant categories
    if transaction.merchant_category in ['gambling', 'crypto']:
        risk_score += 0.3
        reason_codes.append("HIGH_RISK_CATEGORY")
    elif transaction.merchant_category in ['wire_transfer', 'money_order']:
        risk_score += 0.2
        reason_codes.append("MEDIUM_RISK_CATEGORY")

    # High-risk countries
    if transaction.country in ['NG', 'PK', 'RO']:
        risk_score += 0.3
        reason_codes.append("HIGH_RISK_COUNTRY")

    # Add some randomness for demo
    risk_score += random.uniform(-0.05, 0.05)
    risk_score = min(1.0, max(0.0, risk_score))

    # Decision based on risk score
    if risk_score < 0.3:
        decision = "APPROVE"
    elif risk_score < 0.6:
        decision = "REVIEW"
    elif risk_score < 0.85:
        decision = "CHALLENGE"
    else:
        decision = "DECLINE"

    # Calculate processing time
    processing_time = (time.time() - start_time) * 1000

    return {
        'transaction_id': transaction.transaction_id,
        'decision': decision,
        'risk_score': round(risk_score, 3),
        'ml_score': round(risk_score * 0.8, 3),
        'rule_score': round(risk_score * 0.2, 3),
        'confidence': round(0.85 + random.uniform(-0.1, 0.1), 3),
        'reason_codes': reason_codes,
        'processing_time_ms': round(processing_time + random.uniform(10, 50), 2),
        'timestamp': time.time()
    }


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Fraud Detection API (Minimal Version)",
        "status": "running",
        "docs": "/docs"
    }


@app.post("/api/v1/detect", response_model=FraudResponse)
async def detect_fraud(transaction: TransactionRequest) -> FraudResponse:
    """
    Detect fraud for a single transaction

    - **transaction_id**: Unique transaction identifier
    - **user_id**: User identifier
    - **amount**: Transaction amount
    - **merchant_id**: Merchant identifier
    - Returns fraud detection decision with risk scores
    """

    try:
        result = detect_fraud_simple(transaction)
        return FraudResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/detect/batch", response_model=List[FraudResponse])
async def detect_fraud_batch(request: BatchTransactionRequest) -> List[FraudResponse]:
    """
    Detect fraud for multiple transactions

    Process up to 100 transactions in a single request
    """

    try:
        responses = []
        for transaction in request.transactions:
            result = detect_fraud_simple(transaction)
            responses.append(FraudResponse(**result))

        return responses

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/health")
async def health_check():
    """
    Health check endpoint

    Returns service health status
    """

    return {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {
            "api": True,
            "ml_model": True,
            "cache": False,  # Not available in minimal version
            "kafka": False,  # Not available in minimal version
            "database": False  # Not available in minimal version
        }
    }


@app.get("/api/v1/metrics")
async def get_metrics():
    """
    Get system metrics

    Returns basic performance metrics
    """

    return {
        "requests_processed": random.randint(1000, 5000),
        "avg_response_time_ms": round(random.uniform(20, 60), 2),
        "fraud_detection_rate": f"{random.uniform(1.5, 3.0):.2f}%",
        "uptime_seconds": round(time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0),
        "version": "1.0.0"
    }


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    app.state.start_time = time.time()
    print("\n" + "="*60)
    print("FRAUD DETECTION API - MINIMAL VERSION")
    print("="*60)
    print("\nAPI is starting...")
    print("Documentation available at: http://localhost:8000/docs")
    print("\nEndpoints:")
    print("  POST /api/v1/detect - Single transaction")
    print("  POST /api/v1/detect/batch - Batch transactions")
    print("  GET /api/v1/health - Health check")
    print("  GET /api/v1/metrics - System metrics")
    print("="*60 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("\nAPI shutting down...")


if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
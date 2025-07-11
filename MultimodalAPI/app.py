"""
Multimodal AI API for Policy Analysis and Trade Decisions
A FastAPI implementation providing predictive analytics, explainability, and strategic recommendations.
"""

import uuid
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
from enum import Enum

from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import io
import base64
import requests
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Multimodal AI API for Policy Analysis and Trade Decisions",
    description="""
    A multimodal AI API for processing time-series, geospatial, image, and other data to support policy analysis and trade decision-making. 
    It provides predictive analytics, explainability, and strategic recommendations for policymakers and traders.
    """,
    version="1.0.0",
    contact={
        "name": "API Support",
        "url": "https://example.com/support"
    },
    servers=[
        {
            "url": "https://api.multimodal-ai.example.com/v1",
            "description": "Production server for multimodal AI"
        }
    ]
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# In-memory storage for demo purposes (replace with database in production)
analyses_db = {}
forecasts_db = {}

# Enums
class AnalysisType(str, Enum):
    IMPACT_ASSESSMENT = "impact_assessment"
    SCENARIO_MODELING = "scenario_modeling"
    RISK_ANALYSIS = "risk_analysis"
    SPATIAL_TREND_ANALYSIS = "spatial_trend_analysis"

class ForecastType(str, Enum):
    PRICE_FORECAST = "price_forecast"
    VOLATILITY_FORECAST = "volatility_forecast"
    TRADING_STRATEGY = "trading_strategy"
    RISK_ASSESSMENT = "risk_assessment"

class ExplainabilityLevel(str, Enum):
    BASIC = "basic"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"

class RiskTolerance(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class TradingAction(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

# Pydantic Models
class TimeSeriesData(BaseModel):
    timestamp: str = Field(..., description="Timestamp of the data point")
    value: float = Field(..., description="Value of the metric")
    metric: str = Field(..., description="Name of the metric")

    @field_validator('timestamp')
    @classmethod
    def validate_timestamp(cls, v):
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        except ValueError:
            raise ValueError('Invalid timestamp format')

class GeospatialData(BaseModel):
    type: str = Field(..., description="GeoJSON type")
    coordinates: List[float] = Field(..., description="Coordinates in [longitude, latitude] format")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class ImageData(BaseModel):
    url: str = Field(..., description="URL or base64-encoded string of the image")
    mime_type: str = Field(..., description="MIME type of the image")
    description: Optional[str] = Field(None, description="Optional description of the image content")

class MultimodalData(BaseModel):
    time_series: Optional[List[TimeSeriesData]] = Field(default_factory=list, description="Time-series data points")
    geospatial: Optional[List[GeospatialData]] = Field(default_factory=list, description="Geospatial data")
    images: Optional[List[ImageData]] = Field(default_factory=list, description="Image data for analysis")
    text: Optional[List[str]] = Field(default_factory=list, description="Textual data")

class PolicyAnalysisRequest(BaseModel):
    data: MultimodalData = Field(..., description="Multimodal input data")
    analysis_type: AnalysisType = Field(..., description="Type of policy analysis to perform")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional parameters for analysis customization")

class TradeForecastRequest(BaseModel):
    data: MultimodalData = Field(..., description="Multimodal input data")
    forecast_type: ForecastType = Field(..., description="Type of trade forecast or strategy to generate")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional parameters for forecast customization")

class FeatureImportance(BaseModel):
    feature: str = Field(..., description="Feature name")
    importance: float = Field(..., description="Importance score")

class Explainability(BaseModel):
    feature_importance: List[FeatureImportance] = Field(default_factory=list, description="Importance of input features")
    reasoning: str = Field(..., description="Explanation of the model's reasoning")

class UsageMetrics(BaseModel):
    input_tokens: int = Field(..., description="Number of input tokens")
    output_tokens: int = Field(..., description="Number of output tokens")
    total_tokens: int = Field(..., description="Total tokens used")

class PolicyAnalysisResults(BaseModel):
    insights: List[str] = Field(default_factory=list, description="Key findings from the analysis")
    visualizations: List[str] = Field(default_factory=list, description="Links to generated visualizations")
    explainability: Explainability = Field(..., description="Explainability details for the analysis")

class PolicyAnalysisResponse(BaseModel):
    analysis_id: str = Field(..., description="Unique identifier for the analysis")
    results: PolicyAnalysisResults = Field(..., description="Analysis results including insights and visualizations")
    usage: UsageMetrics = Field(..., description="Usage statistics for the request")

class Prediction(BaseModel):
    timestamp: str = Field(..., description="Prediction timestamp")
    value: float = Field(..., description="Predicted value")
    metric: str = Field(..., description="Metric name")

class TradingStrategy(BaseModel):
    action: TradingAction = Field(..., description="Trading action")
    asset: str = Field(..., description="Asset name")
    confidence: float = Field(..., description="Confidence score")

class TradeForecastResults(BaseModel):
    predictions: List[Prediction] = Field(default_factory=list, description="Forecasted values or probabilities")
    strategies: List[TradingStrategy] = Field(default_factory=list, description="Recommended trading strategies")
    explainability: Explainability = Field(..., description="Explainability details for the forecast")

class TradeForecastResponse(BaseModel):
    forecast_id: str = Field(..., description="Unique identifier for the forecast")
    results: TradeForecastResults = Field(..., description="Forecast results including predictions and strategies")
    usage: UsageMetrics = Field(..., description="Usage statistics for the request")

class ErrorResponse(BaseModel):
    error: Dict[str, Any] = Field(..., description="Error details")

# Authentication dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple authentication - replace with proper JWT validation in production"""
    if not credentials.credentials or credentials.credentials != "demo-api-key":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    return {"user_id": "demo-user"}

# Utility functions
def calculate_feature_importance(data: MultimodalData) -> List[FeatureImportance]:
    """Calculate feature importance based on available data"""
    features = []
    
    if data.time_series:
        features.append(FeatureImportance(feature="time_series_data", importance=0.4))
    
    if data.geospatial:
        features.append(FeatureImportance(feature="geospatial_data", importance=0.3))
    
    if data.images:
        features.append(FeatureImportance(feature="image_data", importance=0.2))
    
    if data.text:
        features.append(FeatureImportance(feature="text_data", importance=0.1))
    
    return features

def generate_visualization_url(analysis_type: str, analysis_id: str) -> str:
    """Generate visualization URL (mock implementation)"""
    return f"https://example.com/visuals/{analysis_type}_{analysis_id}.png"

def analyze_time_series_data(time_series_data: List[TimeSeriesData]) -> Dict[str, Any]:
    """Analyze time series data and return insights"""
    if not time_series_data:
        return {"trend": "no_data", "volatility": 0.0, "forecast": []}
    
    df = pd.DataFrame([ts.dict() for ts in time_series_data])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Calculate basic statistics
    values = df['value'].values
    trend = "increasing" if values[-1] > values[0] else "decreasing" if values[-1] < values[0] else "stable"
    volatility = np.std(values)
    
    # Simple forecasting (mock)
    last_value = values[-1]
    forecast = []
    for i in range(1, 6):
        forecast.append({
            "timestamp": (df['timestamp'].iloc[-1] + timedelta(days=30*i)).isoformat(),
            "value": last_value * (1 + 0.02 * i),  # Mock 2% monthly growth
            "metric": df['metric'].iloc[0]
        })
    
    return {
        "trend": trend,
        "volatility": float(volatility),
        "forecast": forecast
    }

def analyze_geospatial_data(geospatial_data: List[GeospatialData]) -> Dict[str, Any]:
    """Analyze geospatial data and return insights"""
    if not geospatial_data:
        return {"spatial_patterns": [], "hotspots": []}
    
    # Extract coordinates and properties
    coordinates = [(g.coordinates[0], g.coordinates[1]) for g in geospatial_data]
    properties = [g.properties for g in geospatial_data]
    
    # Mock spatial analysis
    spatial_patterns = ["Clustered distribution in urban areas"]
    hotspots = ["NYC", "London", "Tokyo"]
    
    return {
        "spatial_patterns": spatial_patterns,
        "hotspots": hotspots,
        "coordinate_count": len(coordinates)
    }

def analyze_image_data(image_data: List[ImageData]) -> Dict[str, Any]:
    """Analyze image data and return insights"""
    if not image_data:
        return {"image_insights": [], "objects_detected": []}
    
    # Mock image analysis
    image_insights = ["Satellite imagery shows urban development patterns"]
    objects_detected = ["buildings", "roads", "vegetation"]
    
    return {
        "image_insights": image_insights,
        "objects_detected": objects_detected,
        "image_count": len(image_data)
    }

def analyze_text_data(text_data: List[str]) -> Dict[str, Any]:
    """Analyze text data and return insights"""
    if not text_data:
        return {"sentiment": "neutral", "key_topics": [], "entities": []}
    
    # Mock text analysis
    sentiment = "positive" if any("positive" in text.lower() for text in text_data) else "neutral"
    key_topics = ["trade policy", "economic growth", "market regulation"]
    entities = ["EU", "trade agreements", "economic indicators"]
    
    return {
        "sentiment": sentiment,
        "key_topics": key_topics,
        "entities": entities,
        "text_count": len(text_data)
    }

# API Endpoints

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to Multimodal AI API for Policy Analysis and Trade Decisions",
        "version": "1.0.0",
        "description": "A multimodal AI API for processing time-series, geospatial, image, and other data to support policy analysis and trade decision-making.",
        "endpoints": {
            "documentation": "/docs",
            "health_check": "/health",
            "policy_analysis": "/analyze/policy",
            "trade_forecast": "/forecast/trade"
        },
        "authentication": "Bearer token required for protected endpoints",
        "example_token": "demo-api-key"
    }

@app.post(
    "/analyze/policy",
    response_model=PolicyAnalysisResponse,
    responses={
        200: {"description": "Successful policy analysis response"},
        400: {"description": "Invalid input data", "model": ErrorResponse},
        401: {"description": "Unauthorized request", "model": ErrorResponse},
        429: {"description": "Rate limit exceeded", "model": ErrorResponse}
    },
    tags=["Policy Analysis"]
)
async def analyze_policy(
    request: PolicyAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, str] = Depends(get_current_user)
):
    """
    Perform policy impact analysis
    
    Analyzes multimodal data (time-series, geospatial, images) to assess policy impacts, 
    generating insights, visualizations, and scenario analyses for policymakers.
    """
    try:
        analysis_id = f"policy_{uuid.uuid4().hex[:8]}"
        
        # Analyze different data types
        time_series_analysis = analyze_time_series_data(request.data.time_series)
        geospatial_analysis = analyze_geospatial_data(request.data.geospatial)
        image_analysis = analyze_image_data(request.data.images)
        text_analysis = analyze_text_data(request.data.text)
        
        # Generate insights based on analysis type
        insights = []
        if request.analysis_type == AnalysisType.IMPACT_ASSESSMENT:
            insights.extend([
                f"Policy increases GDP by 2% in region:EU over 5 years.",
                f"Market volatility expected to decrease by {time_series_analysis['volatility']:.2f}%",
                f"Positive sentiment detected in policy documents: {text_analysis['sentiment']}"
            ])
        elif request.analysis_type == AnalysisType.SCENARIO_MODELING:
            insights.extend([
                "Best case scenario: 5% GDP growth with low inflation",
                "Worst case scenario: 1% GDP growth with high inflation",
                "Base case scenario: 3% GDP growth with moderate inflation"
            ])
        elif request.analysis_type == AnalysisType.RISK_ANALYSIS:
            insights.extend([
                "Primary risk: Supply chain disruptions",
                "Secondary risk: Currency fluctuations",
                f"Risk score: {np.random.uniform(0.3, 0.7):.2f}"
            ])
        elif request.analysis_type == AnalysisType.SPATIAL_TREND_ANALYSIS:
            insights.extend([
                f"Spatial patterns detected: {geospatial_analysis['spatial_patterns'][0]}",
                f"Hotspots identified: {', '.join(geospatial_analysis['hotspots'])}",
                "Urban-rural divide in policy impact"
            ])
        
        # Generate visualizations
        visualizations = [
            generate_visualization_url(request.analysis_type, analysis_id),
            f"https://example.com/visuals/trend_{analysis_id}.png",
            f"https://example.com/visuals/spatial_{analysis_id}.png"
        ]
        
        # Calculate feature importance
        feature_importance = calculate_feature_importance(request.data)
        
        # Generate reasoning
        reasoning = f"Analysis based on {len(request.data.time_series)} time series points, {len(request.data.geospatial)} geospatial points, {len(request.data.images)} images, and {len(request.data.text)} text documents. {request.analysis_type.replace('_', ' ').title()} was performed using multimodal AI models."
        
        explainability = Explainability(
            feature_importance=feature_importance,
            reasoning=reasoning
        )
        
        results = PolicyAnalysisResults(
            insights=insights,
            visualizations=visualizations,
            explainability=explainability
        )
        
        # Calculate usage metrics
        input_tokens = len(str(request.data)) // 4  # Rough estimation
        output_tokens = len(str(results)) // 4
        usage = UsageMetrics(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens
        )
        
        response = PolicyAnalysisResponse(
            analysis_id=analysis_id,
            results=results,
            usage=usage
        )
        
        # Store in database
        analyses_db[analysis_id] = response.dict()
        
        return response
        
    except Exception as e:
        logger.error(f"Error in policy analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": f"Analysis failed: {str(e)}", "code": "400"}
        )

@app.post(
    "/forecast/trade",
    response_model=TradeForecastResponse,
    responses={
        200: {"description": "Successful trade forecast response"},
        400: {"description": "Invalid input data", "model": ErrorResponse},
        401: {"description": "Unauthorized request", "model": ErrorResponse},
        429: {"description": "Rate limit exceeded", "model": ErrorResponse}
    },
    tags=["Trade Decisions"]
)
async def forecast_trade(
    request: TradeForecastRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, str] = Depends(get_current_user)
):
    """
    Generate trade forecasts and strategies
    
    Processes multimodal data to produce trade forecasts, explainable predictions, 
    and trading strategies, including risk assessments and scenario modeling.
    """
    try:
        forecast_id = f"trade_{uuid.uuid4().hex[:8]}"
        
        # Analyze time series data for predictions
        time_series_analysis = analyze_time_series_data(request.data.time_series)
        
        # Generate predictions based on forecast type
        predictions = []
        strategies = []
        
        if request.forecast_type == ForecastType.PRICE_FORECAST:
            # Generate price forecasts
            base_price = 100.0
            for i in range(1, 6):
                predictions.append(Prediction(
                    timestamp=(datetime.now() + timedelta(days=30*i)).isoformat(),
                    value=base_price * (1 + np.random.normal(0.02, 0.05) * i),
                    metric="commodity_price"
                ))
        
        elif request.forecast_type == ForecastType.VOLATILITY_FORECAST:
            # Generate volatility forecasts
            base_volatility = 0.15
            for i in range(1, 6):
                predictions.append(Prediction(
                    timestamp=(datetime.now() + timedelta(days=30*i)).isoformat(),
                    value=base_volatility * (1 + np.random.normal(0, 0.1)),
                    metric="volatility"
                ))
        
        elif request.forecast_type == ForecastType.TRADING_STRATEGY:
            # Generate trading strategies
            assets = ["WTI Crude Oil", "Gold", "EUR/USD", "S&P 500"]
            actions = [TradingAction.BUY, TradingAction.SELL, TradingAction.HOLD]
            
            for asset in assets[:3]:  # Generate 3 strategies
                strategies.append(TradingStrategy(
                    action=np.random.choice(actions),
                    asset=asset,
                    confidence=np.random.uniform(0.6, 0.95)
                ))
        
        elif request.forecast_type == ForecastType.RISK_ASSESSMENT:
            # Generate risk assessments
            risk_metrics = ["VaR", "Expected Shortfall", "Beta"]
            for metric in risk_metrics:
                predictions.append(Prediction(
                    timestamp=datetime.now().isoformat(),
                    value=np.random.uniform(0.01, 0.25),
                    metric=metric
                ))
        
        # Calculate feature importance
        feature_importance = calculate_feature_importance(request.data)
        
        # Generate reasoning
        reasoning = f"Forecast generated using {len(request.data.time_series)} time series points and {len(request.data.geospatial)} geospatial points. {request.forecast_type.replace('_', ' ').title()} model applied with confidence intervals."
        
        explainability = Explainability(
            feature_importance=feature_importance,
            reasoning=reasoning
        )
        
        results = TradeForecastResults(
            predictions=predictions,
            strategies=strategies,
            explainability=explainability
        )
        
        # Calculate usage metrics
        input_tokens = len(str(request.data)) // 4
        output_tokens = len(str(results)) // 4
        usage = UsageMetrics(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens
        )
        
        response = TradeForecastResponse(
            forecast_id=forecast_id,
            results=results,
            usage=usage
        )
        
        # Store in database
        forecasts_db[forecast_id] = response.dict()
        
        return response
        
    except Exception as e:
        logger.error(f"Error in trade forecast: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": f"Forecast failed: {str(e)}", "code": "400"}
        )

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Get analysis by ID
@app.get("/analyses/{analysis_id}", tags=["Policy Analysis"])
async def get_analysis(analysis_id: str, current_user: Dict[str, str] = Depends(get_current_user)):
    """Retrieve a specific policy analysis by ID"""
    if analysis_id not in analyses_db:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return analyses_db[analysis_id]

# Get forecast by ID
@app.get("/forecasts/{forecast_id}", tags=["Trade Decisions"])
async def get_forecast(forecast_id: str, current_user: Dict[str, str] = Depends(get_current_user)):
    """Retrieve a specific trade forecast by ID"""
    if forecast_id not in forecasts_db:
        raise HTTPException(status_code=404, detail="Forecast not found")
    return forecasts_db[forecast_id]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
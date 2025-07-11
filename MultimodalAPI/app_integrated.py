"""
Integrated Multimodal AI API for Policy Analysis and Trade Decisions
FastAPI application with Meta-Transformer backend integration.
"""

import uuid
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from enum import Enum

from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Import backend integration
from backend_integration.analysis_engine import get_policy_engine, get_trade_engine
from backend_integration.meta_transformer_integration import initialize_meta_transformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Multimodal AI API for Policy Analysis and Trade Decisions",
    description="""
    A multimodal AI API for processing time-series, geospatial, image, and other data to support policy analysis and trade decision-making. 
    It provides predictive analytics, explainability, and strategic recommendations for policymakers and traders.
    Powered by Meta-Transformer foundation model.
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
    scenarios: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Policy scenarios")
    risk_breakdown: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Risk breakdown")
    spatial_insights: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Spatial analysis insights")
    temporal_trends: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Temporal trend analysis")
    economic_indicators: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Economic indicators")

class PolicyAnalysisResponse(BaseModel):
    analysis_id: str = Field(..., description="Unique identifier for the analysis")
    results: PolicyAnalysisResults = Field(..., description="Analysis results including insights and visualizations")
    usage: UsageMetrics = Field(..., description="Usage statistics for the request")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Processing metadata")
    data_summary: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Input data summary")

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
    market_analysis: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Market analysis")
    risk_assessment: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Risk assessment")
    technical_analysis: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Technical analysis")
    fundamental_analysis: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Fundamental analysis")
    sentiment_analysis: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Sentiment analysis")

class TradeForecastResponse(BaseModel):
    forecast_id: str = Field(..., description="Unique identifier for the forecast")
    results: TradeForecastResults = Field(..., description="Forecast results including predictions and strategies")
    usage: UsageMetrics = Field(..., description="Usage statistics for the request")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Processing metadata")
    data_summary: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Input data summary")

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

# Initialize Meta-Transformer on startup
@app.on_event("startup")
async def startup_event():
    """Initialize Meta-Transformer on application startup"""
    try:
        logger.info("Initializing Meta-Transformer integration...")
        initialize_meta_transformer(device="cpu")  # Use CPU for demo
        logger.info("Meta-Transformer integration initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Meta-Transformer: {e}")
        logger.warning("Running in fallback mode without Meta-Transformer")

# API Endpoints

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to Multimodal AI API for Policy Analysis and Trade Decisions",
        "version": "1.0.0",
        "description": "A multimodal AI API for processing time-series, geospatial, image, and other data to support policy analysis and trade decision-making.",
        "backend": "Meta-Transformer Foundation Model",
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
    Perform policy impact analysis using Meta-Transformer
    
    Analyzes multimodal data (time-series, geospatial, images) to assess policy impacts, 
    generating insights, visualizations, and scenario analyses for policymakers.
    """
    try:
        analysis_id = f"policy_{uuid.uuid4().hex[:8]}"
        
        # Get policy analysis engine
        policy_engine = get_policy_engine()
        
        # Convert Pydantic models to dictionaries
        data_dict = {
            'time_series': [ts.dict() for ts in request.data.time_series],
            'geospatial': [geo.dict() for geo in request.data.geospatial],
            'images': [img.dict() for img in request.data.images],
            'text': request.data.text
        }
        
        # Perform analysis using Meta-Transformer
        analysis_results = policy_engine.analyze_policy_impact(
            data_dict, 
            request.analysis_type, 
            request.parameters
        )
        
        # Extract results
        insights = analysis_results.get('insights', [])
        visualizations = analysis_results.get('visualizations', [])
        feature_importance = analysis_results.get('feature_importance', [])
        confidence_score = analysis_results.get('confidence_score', 0.0)
        metadata = analysis_results.get('metadata', {})
        data_summary = analysis_results.get('data_summary', {})
        
        # Convert feature importance to Pydantic model
        feature_importance_models = [
            FeatureImportance(feature=fi['feature'], importance=fi['importance'])
            for fi in feature_importance
        ]
        
        # Generate reasoning
        reasoning = f"Analysis performed using Meta-Transformer foundation model on {len(request.data.time_series)} time series points, {len(request.data.geospatial)} geospatial points, {len(request.data.images)} images, and {len(request.data.text)} text documents. {request.analysis_type.replace('_', ' ').title()} analysis completed with confidence score {confidence_score:.2f}."
        
        explainability = Explainability(
            feature_importance=feature_importance_models,
            reasoning=reasoning
        )
        
        # Create results object
        results = PolicyAnalysisResults(
            insights=insights,
            visualizations=visualizations,
            explainability=explainability,
            scenarios=analysis_results.get('scenarios'),
            risk_breakdown=analysis_results.get('risk_breakdown'),
            spatial_insights=analysis_results.get('spatial_insights'),
            temporal_trends=analysis_results.get('temporal_trends'),
            economic_indicators=analysis_results.get('economic_indicators')
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
            usage=usage,
            metadata=metadata,
            data_summary=data_summary
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
    Generate trade forecasts and strategies using Meta-Transformer
    
    Processes multimodal data to produce trade forecasts, explainable predictions, 
    and trading strategies, including risk assessments and scenario modeling.
    """
    try:
        forecast_id = f"trade_{uuid.uuid4().hex[:8]}"
        
        # Get trade forecast engine
        trade_engine = get_trade_engine()
        
        # Convert Pydantic models to dictionaries
        data_dict = {
            'time_series': [ts.dict() for ts in request.data.time_series],
            'geospatial': [geo.dict() for geo in request.data.geospatial],
            'images': [img.dict() for img in request.data.images],
            'text': request.data.text
        }
        
        # Perform forecasting using Meta-Transformer
        forecast_results = trade_engine.forecast_trade(
            data_dict, 
            request.forecast_type, 
            request.parameters
        )
        
        # Extract results
        predictions_data = forecast_results.get('predictions', [])
        strategies_data = forecast_results.get('strategies', [])
        feature_importance = forecast_results.get('feature_importance', [])
        confidence_score = forecast_results.get('confidence_score', 0.0)
        metadata = forecast_results.get('metadata', {})
        data_summary = forecast_results.get('data_summary', {})
        
        # Convert predictions to Pydantic models
        predictions = [
            Prediction(
                timestamp=pred['timestamp'],
                value=pred['value'],
                metric=pred['metric']
            )
            for pred in predictions_data
        ]
        
        # Convert strategies to Pydantic models
        strategies = [
            TradingStrategy(
                action=TradingAction(strategy['action']),
                asset=strategy['asset'],
                confidence=strategy['confidence']
            )
            for strategy in strategies_data
        ]
        
        # Convert feature importance to Pydantic model
        feature_importance_models = [
            FeatureImportance(feature=fi['feature'], importance=fi['importance'])
            for fi in feature_importance
        ]
        
        # Generate reasoning
        reasoning = f"Forecast generated using Meta-Transformer foundation model on {len(request.data.time_series)} time series points, {len(request.data.geospatial)} geospatial points, {len(request.data.images)} images, and {len(request.data.text)} text documents. {request.forecast_type.replace('_', ' ').title()} model applied with confidence score {confidence_score:.2f}."
        
        explainability = Explainability(
            feature_importance=feature_importance_models,
            reasoning=reasoning
        )
        
        # Create results object
        results = TradeForecastResults(
            predictions=predictions,
            strategies=strategies,
            explainability=explainability,
            market_analysis=forecast_results.get('market_analysis'),
            risk_assessment=forecast_results.get('risk_assessment'),
            technical_analysis=forecast_results.get('technical_analysis'),
            fundamental_analysis=forecast_results.get('fundamental_analysis'),
            sentiment_analysis=forecast_results.get('sentiment_analysis')
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
            usage=usage,
            metadata=metadata,
            data_summary=data_summary
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
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "backend": "Meta-Transformer",
        "version": "1.0.0"
    }

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
# Multimodal AI API for Policy Analysis and Trade Decisions

A FastAPI-based multimodal AI API for processing time-series, geospatial, image, and other data to support policy analysis and trade decision-making. The API provides predictive analytics, explainability, and strategic recommendations for policymakers and traders.

## Features

- **Policy Analysis**: Impact assessment, scenario modeling, risk analysis, and spatial trend analysis
- **Trade Forecasting**: Price forecasts, volatility predictions, trading strategies, and risk assessments
- **Multimodal Data Processing**: Time-series, geospatial, image, and text data analysis
- **Explainability**: Feature importance analysis and model reasoning explanations
- **Authentication**: Bearer token-based authentication
- **Comprehensive Documentation**: Auto-generated OpenAPI/Swagger documentation

## Quick Start

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd MultimodalAPI
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

The API will be available at `http://localhost:8000`

### API Documentation

Once the server is running, you can access:
- **Interactive API Documentation**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## API Endpoints

### Policy Analysis

#### POST /analyze/policy
Analyzes multimodal data to assess policy impacts, generating insights, visualizations, and scenario analyses.

**Request Example:**
```json
{
  "data": {
    "time_series": [
      {
        "timestamp": "2025-01-01T00:00:00Z",
        "value": 150.25,
        "metric": "GDP_growth"
      }
    ],
    "geospatial": [
      {
        "type": "Point",
        "coordinates": [-73.935242, 40.730610],
        "properties": {
          "region": "NYC",
          "metric": "population_density",
          "value": 27000
        }
      }
    ],
    "images": [
      {
        "url": "https://example.com/satellite_nyc.jpg",
        "mime_type": "image/jpeg",
        "description": "Satellite image of NYC"
      }
    ],
    "text": [
      "New trade policy announced for EU markets."
    ]
  },
  "analysis_type": "impact_assessment",
  "parameters": {
    "time_horizon": "5y",
    "geospatial_scope": "region:EU",
    "explainability_level": "detailed"
  }
}
```

**Response Example:**
```json
{
  "analysis_id": "policy_123456",
  "results": {
    "insights": [
      "Policy increases GDP by 2% in region:EU over 5 years.",
      "Market volatility expected to decrease by 0.15%"
    ],
    "visualizations": [
      "https://example.com/visuals/impact_assessment_123456.png"
    ],
    "explainability": {
      "feature_importance": [
        {
          "feature": "time_series_data",
          "importance": 0.4
        }
      ],
      "reasoning": "Analysis based on 1 time series points, 1 geospatial points, 1 images, and 1 text documents. Impact Assessment was performed using multimodal AI models."
    }
  },
  "usage": {
    "input_tokens": 125,
    "output_tokens": 50,
    "total_tokens": 175
  }
}
```

### Trade Forecasting

#### POST /forecast/trade
Processes multimodal data to produce trade forecasts, explainable predictions, and trading strategies.

**Request Example:**
```json
{
  "data": {
    "time_series": [
      {
        "timestamp": "2025-01-01T00:00:00Z",
        "value": 100.0,
        "metric": "commodity_price"
      }
    ]
  },
  "forecast_type": "trading_strategy",
  "parameters": {
    "asset_class": "commodities",
    "time_horizon": "1m",
    "risk_tolerance": "medium",
    "explainability_level": "detailed"
  }
}
```

**Response Example:**
```json
{
  "forecast_id": "trade_123456",
  "results": {
    "predictions": [],
    "strategies": [
      {
        "action": "buy",
        "asset": "WTI Crude Oil",
        "confidence": 0.85
      }
    ],
    "explainability": {
      "feature_importance": [
        {
          "feature": "time_series_data",
          "importance": 0.4
        }
      ],
      "reasoning": "Forecast generated using 1 time series points and 0 geospatial points. Trading Strategy model applied with confidence intervals."
    }
  },
  "usage": {
    "input_tokens": 50,
    "output_tokens": 75,
    "total_tokens": 125
  }
}
```

### Additional Endpoints

- **GET /health**: Health check endpoint
- **GET /analyses/{analysis_id}**: Retrieve a specific policy analysis
- **GET /forecasts/{forecast_id}**: Retrieve a specific trade forecast

## Authentication

The API uses Bearer token authentication. Include your API key in the Authorization header:

```
Authorization: Bearer demo-api-key
```

**Note**: For production use, replace the simple authentication with proper JWT validation.

## Data Types

### TimeSeriesData
```json
{
  "timestamp": "2025-01-01T00:00:00Z",
  "value": 150.25,
  "metric": "GDP_growth"
}
```

### GeospatialData
```json
{
  "type": "Point",
  "coordinates": [-73.935242, 40.730610],
  "properties": {
    "region": "NYC",
    "metric": "population_density",
    "value": 27000
  }
}
```

### ImageData
```json
{
  "url": "https://example.com/image.jpg",
  "mime_type": "image/jpeg",
  "description": "Optional description"
}
```

## Analysis Types

### Policy Analysis
- `impact_assessment`: Assess policy impacts on various metrics
- `scenario_modeling`: Generate different policy scenarios
- `risk_analysis`: Analyze risks associated with policies
- `spatial_trend_analysis`: Analyze spatial patterns and trends

### Trade Forecasting
- `price_forecast`: Predict future prices
- `volatility_forecast`: Predict market volatility
- `trading_strategy`: Generate trading recommendations
- `risk_assessment`: Assess trading risks

## Development

### Project Structure
```
MultimodalAPI/
├── app.py              # Main FastAPI application
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── tests/             # Test files (to be added)
```

### Adding New Features

1. **New Analysis Types**: Add to the `AnalysisType` enum and implement corresponding logic
2. **New Data Types**: Extend the `MultimodalData` model and add analysis functions
3. **New Endpoints**: Add new route handlers in `app.py`

### Testing

Run tests (when implemented):
```bash
pytest tests/
```

### Production Deployment

For production deployment:

1. **Database**: Replace in-memory storage with a proper database (PostgreSQL, MongoDB)
2. **Authentication**: Implement proper JWT authentication
3. **Rate Limiting**: Add rate limiting middleware
4. **Monitoring**: Add logging and monitoring (Prometheus, Grafana)
5. **Containerization**: Use Docker for deployment
6. **Load Balancing**: Use a reverse proxy (Nginx)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please contact:
- **API Support**: https://example.com/support
- **Documentation**: http://localhost:8000/docs (when running) 
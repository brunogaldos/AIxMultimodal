#!/usr/bin/env python3
"""
End-to-End Test Suite for Multimodal AI API with Meta-Transformer Backend
Comprehensive testing of policy analysis and trade forecasting endpoints.
"""

import pytest
import requests
import json
import time
from datetime import datetime, timedelta

# Base URL for the API (local development)
BASE_URL = "http://localhost:8000"
API_KEY = "demo-api-key"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Sample multimodal data for testing
SAMPLE_MULTIMODAL_DATA = {
    "time_series": [
        {
            "timestamp": "2025-01-01T00:00:00Z",
            "value": 150.25,
            "metric": "GDP_growth"
        },
        {
            "timestamp": "2025-02-01T00:00:00Z",
            "value": 152.10,
            "metric": "GDP_growth"
        },
        {
            "timestamp": "2025-03-01T00:00:00Z",
            "value": 154.75,
            "metric": "GDP_growth"
        },
        {
            "timestamp": "2025-04-01T00:00:00Z",
            "value": 157.20,
            "metric": "GDP_growth"
        },
        {
            "timestamp": "2025-05-01T00:00:00Z",
            "value": 159.80,
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
        },
        {
            "type": "Point",
            "coordinates": [-0.127758, 51.507351],
            "properties": {
                "region": "London",
                "metric": "population_density",
                "value": 15000
            }
        },
        {
            "type": "Point",
            "coordinates": [139.6917, 35.6895],
            "properties": {
                "region": "Tokyo",
                "metric": "population_density",
                "value": 22000
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
        "New trade policy announced for EU markets with focus on digital transformation.",
        "Economic indicators show positive growth trends across major economies.",
        "Market analysts predict increased investment in renewable energy sectors.",
        "Central banks maintain accommodative monetary policies to support recovery.",
        "Supply chain disruptions easing as global logistics networks adapt."
    ]
}

# Sample trade-specific multimodal data
SAMPLE_TRADE_DATA = {
    "time_series": [
        {
            "timestamp": "2025-01-01T00:00:00Z",
            "value": 75.50,
            "metric": "commodity_price"
        },
        {
            "timestamp": "2025-02-01T00:00:00Z",
            "value": 78.20,
            "metric": "commodity_price"
        },
        {
            "timestamp": "2025-03-01T00:00:00Z",
            "value": 81.10,
            "metric": "commodity_price"
        },
        {
            "timestamp": "2025-04-01T00:00:00Z",
            "value": 84.30,
            "metric": "commodity_price"
        },
        {
            "timestamp": "2025-05-01T00:00:00Z",
            "value": 87.80,
            "metric": "commodity_price"
        }
    ],
    "geospatial": [
        {
            "type": "Point",
            "coordinates": [2.352222, 48.856614],
            "properties": {
                "region": "Paris",
                "metric": "oil_supply",
                "value": 100000
            }
        }
    ],
    "images": [
        {
            "url": "https://example.com/oil_chart.jpg",
            "mime_type": "image/jpeg",
            "description": "Oil price trend chart"
        }
    ],
    "text": [
        "Geopolitical tensions in oil-producing region affecting supply.",
        "Increased demand for renewable energy sources driving commodity prices.",
        "Federal Reserve announces new monetary policy framework.",
        "Trade tensions easing between major economies.",
        "Technological innovations in energy sector creating new opportunities."
    ]
}

class TestAPIHealth:
    """Test API health and basic functionality."""
    
    def test_health_check(self):
        """Test the health check endpoint."""
        response = requests.get(f"{BASE_URL}/health")
        assert response.status_code == 200, f"Expected status 200, got {response.status_code}"
        data = response.json()
        assert "status" in data, "Response missing status"
        assert data["status"] == "healthy", f"Expected 'healthy', got {data['status']}"
        assert "timestamp" in data, "Response missing timestamp"
        assert "backend" in data, "Response missing backend info"
        print(f"✅ Health check passed: {data}")

    def test_root_endpoint(self):
        """Test the root endpoint."""
        response = requests.get(f"{BASE_URL}/")
        assert response.status_code == 200, f"Expected status 200, got {response.status_code}"
        data = response.json()
        assert "message" in data, "Response missing message"
        assert "version" in data, "Response missing version"
        assert "endpoints" in data, "Response missing endpoints"
        print(f"✅ Root endpoint passed: {data['message']}")

class TestPolicyAnalysis:
    """Test policy analysis endpoints."""
    
    def test_policy_analysis_impact_assessment(self):
        """Test successful policy impact assessment with valid multimodal data."""
        payload = {
            "data": SAMPLE_MULTIMODAL_DATA,
            "analysis_type": "impact_assessment",
            "parameters": {
                "time_horizon": "5y",
                "geospatial_scope": "region:EU",
                "explainability_level": "detailed"
            }
        }
        
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/analyze/policy", headers=HEADERS, json=payload)
        end_time = time.time()
        
        assert response.status_code == 200, f"Expected status 200, got {response.status_code}"
        data = response.json()
        
        # Validate response structure
        assert "analysis_id" in data, "Response missing analysis_id"
        assert data["analysis_id"].startswith("policy_"), "Invalid analysis_id format"
        assert "results" in data, "Response missing results"
        assert "insights" in data["results"], "Response missing insights"
        assert isinstance(data["results"]["insights"], list), "Insights should be a list"
        assert len(data["results"]["insights"]) > 0, "No insights generated"
        
        # Validate explainability
        assert "explainability" in data["results"], "Response missing explainability"
        assert "feature_importance" in data["results"]["explainability"], "Response missing feature_importance"
        assert "reasoning" in data["results"]["explainability"], "Response missing reasoning"
        
        # Validate usage metrics
        assert "usage" in data, "Response missing usage"
        assert all(key in data["usage"] for key in ["input_tokens", "output_tokens", "total_tokens"]), "Usage metrics incomplete"
        
        # Validate enhanced features
        if "scenarios" in data["results"]:
            assert isinstance(data["results"]["scenarios"], list), "Scenarios should be a list"
        if "risk_breakdown" in data["results"]:
            assert isinstance(data["results"]["risk_breakdown"], dict), "Risk breakdown should be a dict"
        if "economic_indicators" in data["results"]:
            assert isinstance(data["results"]["economic_indicators"], dict), "Economic indicators should be a dict"
        
        print(f"✅ Policy impact assessment passed in {end_time - start_time:.2f}s")
        print(f"   Analysis ID: {data['analysis_id']}")
        print(f"   Insights: {len(data['results']['insights'])} generated")
        print(f"   Usage: {data['usage']['total_tokens']} tokens")
        
        return data["analysis_id"]

    def test_policy_analysis_scenario_modeling(self):
        """Test policy scenario modeling."""
        payload = {
            "data": SAMPLE_MULTIMODAL_DATA,
            "analysis_type": "scenario_modeling",
            "parameters": {
                "time_horizon": "3y",
                "scenario_count": 3,
                "explainability_level": "detailed"
            }
        }
        
        response = requests.post(f"{BASE_URL}/analyze/policy", headers=HEADERS, json=payload)
        assert response.status_code == 200, f"Expected status 200, got {response.status_code}"
        data = response.json()
        
        assert "results" in data, "Response missing results"
        if "scenarios" in data["results"]:
            assert len(data["results"]["scenarios"]) > 0, "No scenarios generated"
            for scenario in data["results"]["scenarios"]:
                assert "name" in scenario, "Scenario missing name"
                assert "probability" in scenario, "Scenario missing probability"
        
        print(f"✅ Policy scenario modeling passed")
        return data["analysis_id"]

    def test_policy_analysis_risk_analysis(self):
        """Test policy risk analysis."""
        payload = {
            "data": SAMPLE_MULTIMODAL_DATA,
            "analysis_type": "risk_analysis",
            "parameters": {
                "risk_categories": ["supply_chain", "currency", "regulatory"],
                "explainability_level": "detailed"
            }
        }
        
        response = requests.post(f"{BASE_URL}/analyze/policy", headers=HEADERS, json=payload)
        assert response.status_code == 200, f"Expected status 200, got {response.status_code}"
        data = response.json()
        
        assert "results" in data, "Response missing results"
        if "risk_breakdown" in data["results"]:
            assert isinstance(data["results"]["risk_breakdown"], dict), "Risk breakdown should be a dict"
            assert len(data["results"]["risk_breakdown"]) > 0, "No risk categories found"
        
        print(f"✅ Policy risk analysis passed")
        return data["analysis_id"]

    def test_policy_analysis_spatial_trend_analysis(self):
        """Test policy spatial trend analysis."""
        payload = {
            "data": SAMPLE_MULTIMODAL_DATA,
            "analysis_type": "spatial_trend_analysis",
            "parameters": {
                "spatial_resolution": "city",
                "explainability_level": "detailed"
            }
        }
        
        response = requests.post(f"{BASE_URL}/analyze/policy", headers=HEADERS, json=payload)
        assert response.status_code == 200, f"Expected status 200, got {response.status_code}"
        data = response.json()
        
        assert "results" in data, "Response missing results"
        if "spatial_insights" in data["results"]:
            assert isinstance(data["results"]["spatial_insights"], dict), "Spatial insights should be a dict"
        
        print(f"✅ Policy spatial trend analysis passed")
        return data["analysis_id"]

    def test_policy_analysis_invalid_analysis_type(self):
        """Test policy analysis with invalid analysis type."""
        payload = {
            "data": SAMPLE_MULTIMODAL_DATA,
            "analysis_type": "invalid_type",
            "parameters": {
                "time_horizon": "5y",
                "geospatial_scope": "region:EU",
                "explainability_level": "detailed"
            }
        }
        response = requests.post(f"{BASE_URL}/analyze/policy", headers=HEADERS, json=payload)
        
        assert response.status_code == 400, f"Expected status 400, got {response.status_code}"
        data = response.json()
        assert "detail" in data, "Response missing error detail"
        print(f"✅ Invalid analysis type error handling passed")

    def test_policy_analysis_missing_data(self):
        """Test policy analysis with missing required data field."""
        payload = {
            "analysis_type": "impact_assessment",
            "parameters": {
                "time_horizon": "5y",
                "geospatial_scope": "region:EU",
                "explainability_level": "detailed"
            }
        }
        response = requests.post(f"{BASE_URL}/analyze/policy", headers=HEADERS, json=payload)
        
        assert response.status_code == 422, f"Expected status 422, got {response.status_code}"
        print(f"✅ Missing data validation passed")

    def test_policy_analysis_invalid_time_series(self):
        """Test policy analysis with invalid time-series data."""
        invalid_data = SAMPLE_MULTIMODAL_DATA.copy()
        invalid_data["time_series"] = [
            {
                "timestamp": "invalid_date",  # Invalid timestamp format
                "value": 150.25,
                "metric": "GDP_growth"
            }
        ]
        payload = {
            "data": invalid_data,
            "analysis_type": "impact_assessment",
            "parameters": {
                "time_horizon": "5y",
                "geospatial_scope": "region:EU",
                "explainability_level": "detailed"
            }
        }
        response = requests.post(f"{BASE_URL}/analyze/policy", headers=HEADERS, json=payload)
        
        assert response.status_code == 400, f"Expected status 400, got {response.status_code}"
        print(f"✅ Invalid time-series validation passed")

class TestTradeForecasting:
    """Test trade forecasting endpoints."""
    
    def test_trade_forecast_trading_strategy(self):
        """Test successful trading strategy generation with valid multimodal data."""
        payload = {
            "data": SAMPLE_TRADE_DATA,
            "forecast_type": "trading_strategy",
            "parameters": {
                "asset_class": "commodities",
                "time_horizon": "1m",
                "risk_tolerance": "medium",
                "explainability_level": "detailed"
            }
        }
        
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/forecast/trade", headers=HEADERS, json=payload)
        end_time = time.time()
        
        assert response.status_code == 200, f"Expected status 200, got {response.status_code}"
        data = response.json()
        
        # Validate response structure
        assert "forecast_id" in data, "Response missing forecast_id"
        assert data["forecast_id"].startswith("trade_"), "Invalid forecast_id format"
        assert "results" in data, "Response missing results"
        assert "predictions" in data["results"], "Response missing predictions"
        assert isinstance(data["results"]["predictions"], list), "Predictions should be a list"
        assert "strategies" in data["results"], "Response missing strategies"
        assert isinstance(data["results"]["strategies"], list), "Strategies should be a list"
        
        # Validate strategies
        if len(data["results"]["strategies"]) > 0:
            for strategy in data["results"]["strategies"]:
                assert "action" in strategy, "Strategy missing action field"
                assert "asset" in strategy, "Strategy missing asset field"
                assert "confidence" in strategy, "Strategy missing confidence field"
                assert strategy["action"] in ["buy", "sell", "hold"], "Invalid action"
                assert 0 <= strategy["confidence"] <= 1, "Invalid confidence score"
        
        # Validate explainability
        assert "explainability" in data["results"], "Response missing explainability"
        assert "feature_importance" in data["results"]["explainability"], "Response missing feature_importance"
        assert "reasoning" in data["results"]["explainability"], "Response missing reasoning"
        
        # Validate usage metrics
        assert "usage" in data, "Response missing usage"
        assert all(key in data["usage"] for key in ["input_tokens", "output_tokens", "total_tokens"]), "Usage metrics incomplete"
        
        # Validate enhanced features
        if "market_analysis" in data["results"]:
            assert isinstance(data["results"]["market_analysis"], dict), "Market analysis should be a dict"
        if "technical_analysis" in data["results"]:
            assert isinstance(data["results"]["technical_analysis"], dict), "Technical analysis should be a dict"
        if "sentiment_analysis" in data["results"]:
            assert isinstance(data["results"]["sentiment_analysis"], dict), "Sentiment analysis should be a dict"
        
        print(f"✅ Trade strategy generation passed in {end_time - start_time:.2f}s")
        print(f"   Forecast ID: {data['forecast_id']}")
        print(f"   Predictions: {len(data['results']['predictions'])} generated")
        print(f"   Strategies: {len(data['results']['strategies'])} generated")
        print(f"   Usage: {data['usage']['total_tokens']} tokens")
        
        return data["forecast_id"]

    def test_trade_forecast_price_forecast(self):
        """Test price forecasting."""
        payload = {
            "data": SAMPLE_TRADE_DATA,
            "forecast_type": "price_forecast",
            "parameters": {
                "forecast_periods": 5,
                "confidence_interval": 0.95,
                "explainability_level": "detailed"
            }
        }
        
        response = requests.post(f"{BASE_URL}/forecast/trade", headers=HEADERS, json=payload)
        assert response.status_code == 200, f"Expected status 200, got {response.status_code}"
        data = response.json()
        
        assert "results" in data, "Response missing results"
        assert "predictions" in data["results"], "Response missing predictions"
        if len(data["results"]["predictions"]) > 0:
            for prediction in data["results"]["predictions"]:
                assert "timestamp" in prediction, "Prediction missing timestamp"
                assert "value" in prediction, "Prediction missing value"
                assert "metric" in prediction, "Prediction missing metric"
        
        print(f"✅ Price forecasting passed")
        return data["forecast_id"]

    def test_trade_forecast_volatility_forecast(self):
        """Test volatility forecasting."""
        payload = {
            "data": SAMPLE_TRADE_DATA,
            "forecast_type": "volatility_forecast",
            "parameters": {
                "volatility_window": 30,
                "explainability_level": "detailed"
            }
        }
        
        response = requests.post(f"{BASE_URL}/forecast/trade", headers=HEADERS, json=payload)
        assert response.status_code == 200, f"Expected status 200, got {response.status_code}"
        data = response.json()
        
        assert "results" in data, "Response missing results"
        assert "predictions" in data["results"], "Response missing predictions"
        
        print(f"✅ Volatility forecasting passed")
        return data["forecast_id"]

    def test_trade_forecast_risk_assessment(self):
        """Test trade risk assessment."""
        payload = {
            "data": SAMPLE_TRADE_DATA,
            "forecast_type": "risk_assessment",
            "parameters": {
                "risk_metrics": ["var", "cvar", "volatility"],
                "explainability_level": "detailed"
            }
        }
        
        response = requests.post(f"{BASE_URL}/forecast/trade", headers=HEADERS, json=payload)
        assert response.status_code == 200, f"Expected status 200, got {response.status_code}"
        data = response.json()
        
        assert "results" in data, "Response missing results"
        if "risk_assessment" in data["results"]:
            assert isinstance(data["results"]["risk_assessment"], dict), "Risk assessment should be a dict"
        
        print(f"✅ Trade risk assessment passed")
        return data["forecast_id"]

    def test_trade_forecast_invalid_forecast_type(self):
        """Test trade forecast with invalid forecast type."""
        payload = {
            "data": SAMPLE_TRADE_DATA,
            "forecast_type": "invalid_type",
            "parameters": {
                "asset_class": "commodities",
                "time_horizon": "1m",
                "risk_tolerance": "medium",
                "explainability_level": "detailed"
            }
        }
        response = requests.post(f"{BASE_URL}/forecast/trade", headers=HEADERS, json=payload)
        
        assert response.status_code == 400, f"Expected status 400, got {response.status_code}"
        data = response.json()
        assert "detail" in data, "Response missing error detail"
        print(f"✅ Invalid forecast type error handling passed")

class TestRetrievalEndpoints:
    """Test retrieval endpoints."""
    
    def test_retrieve_policy_analysis(self, policy_analysis_id):
        """Test retrieving a policy analysis by ID."""
        if not policy_analysis_id:
            pytest.skip("No policy analysis ID available")
        
        response = requests.get(f"{BASE_URL}/analyses/{policy_analysis_id}", headers=HEADERS)
        assert response.status_code == 200, f"Expected status 200, got {response.status_code}"
        data = response.json()
        
        assert "analysis_id" in data, "Response missing analysis_id"
        assert data["analysis_id"] == policy_analysis_id, "Analysis ID mismatch"
        assert "results" in data, "Response missing results"
        print(f"✅ Policy analysis retrieval passed for ID: {policy_analysis_id}")

    def test_retrieve_trade_forecast(self, trade_forecast_id):
        """Test retrieving a trade forecast by ID."""
        if not trade_forecast_id:
            pytest.skip("No trade forecast ID available")
        
        response = requests.get(f"{BASE_URL}/forecasts/{trade_forecast_id}", headers=HEADERS)
        assert response.status_code == 200, f"Expected status 200, got {response.status_code}"
        data = response.json()
        
        assert "forecast_id" in data, "Response missing forecast_id"
        assert data["forecast_id"] == trade_forecast_id, "Forecast ID mismatch"
        assert "results" in data, "Response missing results"
        print(f"✅ Trade forecast retrieval passed for ID: {trade_forecast_id}")

    def test_retrieve_nonexistent_analysis(self):
        """Test retrieving a nonexistent policy analysis."""
        response = requests.get(f"{BASE_URL}/analyses/nonexistent_id", headers=HEADERS)
        assert response.status_code == 404, f"Expected status 404, got {response.status_code}"
        print(f"✅ Nonexistent analysis retrieval error handling passed")

    def test_retrieve_nonexistent_forecast(self):
        """Test retrieving a nonexistent trade forecast."""
        response = requests.get(f"{BASE_URL}/forecasts/nonexistent_id", headers=HEADERS)
        assert response.status_code == 404, f"Expected status 404, got {response.status_code}"
        print(f"✅ Nonexistent forecast retrieval error handling passed")

class TestAuthentication:
    """Test authentication and authorization."""
    
    def test_unauthorized_request(self):
        """Test request without authentication."""
        payload = {
            "data": SAMPLE_MULTIMODAL_DATA,
            "analysis_type": "impact_assessment"
        }
        response = requests.post(f"{BASE_URL}/analyze/policy", json=payload)
        assert response.status_code == 401, f"Expected status 401, got {response.status_code}"
        print(f"✅ Unauthorized request handling passed")

    def test_invalid_token(self):
        """Test request with invalid token."""
        invalid_headers = {
            "Authorization": "Bearer invalid-token",
            "Content-Type": "application/json"
        }
        payload = {
            "data": SAMPLE_MULTIMODAL_DATA,
            "analysis_type": "impact_assessment"
        }
        response = requests.post(f"{BASE_URL}/analyze/policy", headers=invalid_headers, json=payload)
        assert response.status_code == 401, f"Expected status 401, got {response.status_code}"
        print(f"✅ Invalid token handling passed")

class TestPerformance:
    """Test API performance and response times."""
    
    def test_response_time_policy_analysis(self):
        """Test response time for policy analysis."""
        payload = {
            "data": SAMPLE_MULTIMODAL_DATA,
            "analysis_type": "impact_assessment",
            "parameters": {
                "time_horizon": "1y",
                "explainability_level": "basic"
            }
        }
        
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/analyze/policy", headers=HEADERS, json=payload)
        end_time = time.time()
        
        assert response.status_code == 200, f"Expected status 200, got {response.status_code}"
        response_time = end_time - start_time
        
        # Performance expectations
        assert response_time < 10.0, f"Response time {response_time:.2f}s exceeds 10s limit"
        print(f"✅ Policy analysis response time: {response_time:.2f}s")

    def test_response_time_trade_forecast(self):
        """Test response time for trade forecasting."""
        payload = {
            "data": SAMPLE_TRADE_DATA,
            "forecast_type": "trading_strategy",
            "parameters": {
                "time_horizon": "1m",
                "risk_tolerance": "medium"
            }
        }
        
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/forecast/trade", headers=HEADERS, json=payload)
        end_time = time.time()
        
        assert response.status_code == 200, f"Expected status 200, got {response.status_code}"
        response_time = end_time - start_time
        
        # Performance expectations
        assert response_time < 10.0, f"Response time {response_time:.2f}s exceeds 10s limit"
        print(f"✅ Trade forecast response time: {response_time:.2f}s")

def main():
    """Run the E2E test suite."""
    print("=" * 80)
    print("🚀 Multimodal AI API E2E Test Suite with Meta-Transformer Backend")
    print("=" * 80)
    
    # Check if API is running
    try:
        health_response = requests.get(f"{BASE_URL}/health", timeout=5)
        if health_response.status_code != 200:
            print("❌ API is not running or not responding. Please start the API first.")
            print("   Run: python app_integrated.py or ./start_integrated.sh")
            return
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to API. Please ensure it's running on localhost:8000")
        print("   Run: python app_integrated.py or ./start_integrated.sh")
        return
    
    print("✅ API is running and responding")
    print()
    
    # Run tests with pytest
    import sys
    import subprocess
    
    # Run pytest with verbose output
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        __file__, 
        "-v", 
        "--tb=short",
        "--capture=no"
    ])
    
    print()
    print("=" * 80)
    if result.returncode == 0:
        print("🎉 All E2E tests passed successfully!")
        print("✅ Multimodal AI API with Meta-Transformer backend is working correctly")
    else:
        print("❌ Some E2E tests failed. Please check the output above.")
    print("=" * 80)

if __name__ == "__main__":
    main() 
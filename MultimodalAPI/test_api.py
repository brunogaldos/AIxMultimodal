#!/usr/bin/env python3
"""
Test script for the Multimodal AI API
Demonstrates how to use the API endpoints with sample data.
"""

import requests
import json
from datetime import datetime, timedelta

# API Configuration
BASE_URL = "http://localhost:8000"
API_KEY = "demo-api-key"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_policy_analysis():
    """Test policy analysis endpoint"""
    print("Testing policy analysis...")
    
    # Sample data
    request_data = {
        "data": {
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
                "New trade policy announced for EU markets.",
                "Economic indicators show positive growth trends.",
                "Market analysts predict increased investment in renewable energy."
            ]
        },
        "analysis_type": "impact_assessment",
        "parameters": {
            "time_horizon": "5y",
            "geospatial_scope": "region:EU",
            "explainability_level": "detailed"
        }
    }
    
    response = requests.post(
        f"{BASE_URL}/analyze/policy",
        headers=HEADERS,
        json=request_data
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Analysis ID: {result['analysis_id']}")
        print(f"Insights: {result['results']['insights']}")
        print(f"Visualizations: {result['results']['visualizations']}")
        print(f"Usage: {result['usage']}")
        
        # Store analysis ID for later retrieval
        return result['analysis_id']
    else:
        print(f"Error: {response.text}")
        return None

def test_trade_forecast():
    """Test trade forecast endpoint"""
    print("Testing trade forecast...")
    
    # Sample data
    request_data = {
        "data": {
            "time_series": [
                {
                    "timestamp": "2025-01-01T00:00:00Z",
                    "value": 100.0,
                    "metric": "commodity_price"
                },
                {
                    "timestamp": "2025-02-01T00:00:00Z",
                    "value": 102.5,
                    "metric": "commodity_price"
                },
                {
                    "timestamp": "2025-03-01T00:00:00Z",
                    "value": 105.2,
                    "metric": "commodity_price"
                },
                {
                    "timestamp": "2025-04-01T00:00:00Z",
                    "value": 108.1,
                    "metric": "commodity_price"
                }
            ],
            "geospatial": [
                {
                    "type": "Point",
                    "coordinates": [-74.006, 40.7128],
                    "properties": {
                        "region": "NYC",
                        "metric": "trading_volume",
                        "value": 1000000
                    }
                }
            ],
            "text": [
                "Oil supply disruptions in Middle East",
                "Increased demand for renewable energy sources",
                "Federal Reserve announces new monetary policy"
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
    
    response = requests.post(
        f"{BASE_URL}/forecast/trade",
        headers=HEADERS,
        json=request_data
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Forecast ID: {result['forecast_id']}")
        print(f"Predictions: {result['results']['predictions']}")
        print(f"Strategies: {result['results']['strategies']}")
        print(f"Usage: {result['usage']}")
        
        # Store forecast ID for later retrieval
        return result['forecast_id']
    else:
        print(f"Error: {response.text}")
        return None

def test_get_analysis(analysis_id):
    """Test retrieving a specific analysis"""
    if not analysis_id:
        return
    
    print(f"Testing get analysis for ID: {analysis_id}")
    response = requests.get(
        f"{BASE_URL}/analyses/{analysis_id}",
        headers=HEADERS
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Retrieved analysis: {result['analysis_id']}")
    else:
        print(f"Error: {response.text}")
    print()

def test_get_forecast(forecast_id):
    """Test retrieving a specific forecast"""
    if not forecast_id:
        return
    
    print(f"Testing get forecast for ID: {forecast_id}")
    response = requests.get(
        f"{BASE_URL}/forecasts/{forecast_id}",
        headers=HEADERS
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Retrieved forecast: {result['forecast_id']}")
    else:
        print(f"Error: {response.text}")
    print()

def test_different_analysis_types():
    """Test different analysis types"""
    print("Testing different analysis types...")
    
    analysis_types = [
        "impact_assessment",
        "scenario_modeling", 
        "risk_analysis",
        "spatial_trend_analysis"
    ]
    
    base_data = {
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
            ]
        },
        "parameters": {
            "time_horizon": "1y",
            "explainability_level": "basic"
        }
    }
    
    for analysis_type in analysis_types:
        print(f"\nTesting {analysis_type}...")
        request_data = base_data.copy()
        request_data["analysis_type"] = analysis_type
        
        response = requests.post(
            f"{BASE_URL}/analyze/policy",
            headers=HEADERS,
            json=request_data
        )
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Analysis ID: {result['analysis_id']}")
            print(f"First insight: {result['results']['insights'][0] if result['results']['insights'] else 'No insights'}")
        else:
            print(f"Error: {response.text}")

def test_different_forecast_types():
    """Test different forecast types"""
    print("Testing different forecast types...")
    
    forecast_types = [
        "price_forecast",
        "volatility_forecast",
        "trading_strategy",
        "risk_assessment"
    ]
    
    base_data = {
        "data": {
            "time_series": [
                {
                    "timestamp": "2025-01-01T00:00:00Z",
                    "value": 100.0,
                    "metric": "commodity_price"
                }
            ]
        },
        "parameters": {
            "time_horizon": "1m",
            "risk_tolerance": "medium",
            "explainability_level": "basic"
        }
    }
    
    for forecast_type in forecast_types:
        print(f"\nTesting {forecast_type}...")
        request_data = base_data.copy()
        request_data["forecast_type"] = forecast_type
        
        response = requests.post(
            f"{BASE_URL}/forecast/trade",
            headers=HEADERS,
            json=request_data
        )
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Forecast ID: {result['forecast_id']}")
            if result['results']['predictions']:
                print(f"First prediction: {result['results']['predictions'][0]}")
            if result['results']['strategies']:
                print(f"First strategy: {result['results']['strategies'][0]}")
        else:
            print(f"Error: {response.text}")

def test_error_handling():
    """Test error handling"""
    print("Testing error handling...")
    
    # Test without authentication
    print("Testing without authentication...")
    response = requests.post(
        f"{BASE_URL}/analyze/policy",
        json={"data": {}, "analysis_type": "impact_assessment"}
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    print()
    
    # Test invalid data
    print("Testing invalid data...")
    response = requests.post(
        f"{BASE_URL}/analyze/policy",
        headers=HEADERS,
        json={"data": {}, "analysis_type": "invalid_type"}
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    print()

def main():
    """Run all tests"""
    print("=" * 60)
    print("Multimodal AI API Test Suite")
    print("=" * 60)
    
    # Test health check
    test_health_check()
    
    # Test policy analysis
    analysis_id = test_policy_analysis()
    
    # Test trade forecast
    forecast_id = test_trade_forecast()
    
    # Test retrieval endpoints
    test_get_analysis(analysis_id)
    test_get_forecast(forecast_id)
    
    # Test different analysis types
    test_different_analysis_types()
    
    # Test different forecast types
    test_different_forecast_types()
    
    # Test error handling
    test_error_handling()
    
    print("=" * 60)
    print("Test suite completed!")
    print("=" * 60)

if __name__ == "__main__":
    main() 
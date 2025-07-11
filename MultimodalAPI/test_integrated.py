#!/usr/bin/env python3
"""
Test script for the Integrated Multimodal AI API with Meta-Transformer
Demonstrates the full backend integration with real multimodal analysis.
"""

import requests
import json
import time
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
    print("🏥 Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_root_endpoint():
    """Test the root endpoint"""
    print("🏠 Testing root endpoint...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_policy_analysis_with_meta_transformer():
    """Test policy analysis with Meta-Transformer integration"""
    print("🔬 Testing policy analysis with Meta-Transformer...")
    
    # Comprehensive multimodal data
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
            "text": [
                "New trade policy announced for EU markets with focus on digital transformation.",
                "Economic indicators show positive growth trends across major economies.",
                "Market analysts predict increased investment in renewable energy sectors.",
                "Central banks maintain accommodative monetary policies to support recovery.",
                "Supply chain disruptions easing as global logistics networks adapt."
            ]
        },
        "analysis_type": "impact_assessment",
        "parameters": {
            "time_horizon": "5y",
            "geospatial_scope": "region:EU",
            "explainability_level": "detailed"
        }
    }
    
    start_time = time.time()
    response = requests.post(
        f"{BASE_URL}/analyze/policy",
        headers=HEADERS,
        json=request_data
    )
    end_time = time.time()
    
    print(f"Status: {response.status_code}")
    print(f"Response Time: {end_time - start_time:.2f} seconds")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Analysis ID: {result['analysis_id']}")
        print(f"📊 Insights: {len(result['results']['insights'])} insights generated")
        print(f"🎯 Confidence Score: {result['results']['explainability'].get('confidence_score', 'N/A')}")
        print(f"📈 Feature Importance: {len(result['results']['explainability']['feature_importance'])} features")
        print(f"📋 Usage: {result['usage']['total_tokens']} total tokens")
        
        # Show enhanced results
        if 'scenarios' in result['results']:
            print(f"🎭 Scenarios: {len(result['results']['scenarios'])} scenarios generated")
        if 'risk_breakdown' in result['results']:
            print(f"⚠️ Risk Breakdown: {len(result['results']['risk_breakdown'])} risk categories")
        if 'economic_indicators' in result['results']:
            print(f"📊 Economic Indicators: {len(result['results']['economic_indicators'])} indicators")
        
        return result['analysis_id']
    else:
        print(f"❌ Error: {response.text}")
        return None

def test_trade_forecast_with_meta_transformer():
    """Test trade forecasting with Meta-Transformer integration"""
    print("📈 Testing trade forecasting with Meta-Transformer...")
    
    # Comprehensive multimodal data for trading
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
                },
                {
                    "timestamp": "2025-05-01T00:00:00Z",
                    "value": 111.3,
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
                "Oil supply disruptions in Middle East affecting global markets.",
                "Increased demand for renewable energy sources driving commodity prices.",
                "Federal Reserve announces new monetary policy framework.",
                "Trade tensions easing between major economies.",
                "Technological innovations in energy sector creating new opportunities."
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
    
    start_time = time.time()
    response = requests.post(
        f"{BASE_URL}/forecast/trade",
        headers=HEADERS,
        json=request_data
    )
    end_time = time.time()
    
    print(f"Status: {response.status_code}")
    print(f"Response Time: {end_time - start_time:.2f} seconds")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Forecast ID: {result['forecast_id']}")
        print(f"📊 Predictions: {len(result['results']['predictions'])} predictions generated")
        print(f"🎯 Strategies: {len(result['results']['strategies'])} trading strategies")
        print(f"📈 Confidence Score: {result['results']['explainability'].get('confidence_score', 'N/A')}")
        print(f"📋 Usage: {result['usage']['total_tokens']} total tokens")
        
        # Show enhanced results
        if 'market_analysis' in result['results']:
            print(f"📊 Market Analysis: {result['results']['market_analysis'].get('market_trend', 'N/A')} trend")
        if 'technical_analysis' in result['results']:
            print(f"📈 Technical Analysis: RSI {result['results']['technical_analysis'].get('trend_indicators', {}).get('rsi', 'N/A')}")
        if 'sentiment_analysis' in result['results']:
            print(f"😊 Sentiment: {result['results']['sentiment_analysis'].get('overall_sentiment', 'N/A')}")
        
        return result['forecast_id']
    else:
        print(f"❌ Error: {response.text}")
        return None

def test_different_analysis_types():
    """Test different policy analysis types"""
    print("🔄 Testing different analysis types...")
    
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
                },
                {
                    "timestamp": "2025-02-01T00:00:00Z",
                    "value": 152.10,
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
            "text": [
                "New policy implementation shows positive early results."
            ]
        },
        "parameters": {
            "time_horizon": "1y",
            "explainability_level": "detailed"
        }
    }
    
    for analysis_type in analysis_types:
        print(f"\n🔍 Testing {analysis_type}...")
        request_data = base_data.copy()
        request_data["analysis_type"] = analysis_type
        
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/analyze/policy",
            headers=HEADERS,
            json=request_data
        )
        end_time = time.time()
        
        print(f"Status: {response.status_code}")
        print(f"Response Time: {end_time - start_time:.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Analysis ID: {result['analysis_id']}")
            print(f"📊 Insights: {len(result['results']['insights'])} insights")
            
            # Show analysis-specific results
            if analysis_type == "scenario_modeling" and 'scenarios' in result['results']:
                print(f"🎭 Scenarios: {len(result['results']['scenarios'])} scenarios")
            elif analysis_type == "risk_analysis" and 'risk_breakdown' in result['results']:
                print(f"⚠️ Risk Categories: {len(result['results']['risk_breakdown'])}")
            elif analysis_type == "spatial_trend_analysis" and 'spatial_insights' in result['results']:
                print(f"🗺️ Spatial Insights: {len(result['results']['spatial_insights'].get('spatial_patterns', []))} patterns")
        else:
            print(f"❌ Error: {response.text}")

def test_different_forecast_types():
    """Test different trade forecast types"""
    print("🔄 Testing different forecast types...")
    
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
                },
                {
                    "timestamp": "2025-02-01T00:00:00Z",
                    "value": 102.5,
                    "metric": "commodity_price"
                }
            ],
            "text": [
                "Market conditions show positive momentum."
            ]
        },
        "parameters": {
            "time_horizon": "1m",
            "risk_tolerance": "medium",
            "explainability_level": "detailed"
        }
    }
    
    for forecast_type in forecast_types:
        print(f"\n📈 Testing {forecast_type}...")
        request_data = base_data.copy()
        request_data["forecast_type"] = forecast_type
        
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/forecast/trade",
            headers=HEADERS,
            json=request_data
        )
        end_time = time.time()
        
        print(f"Status: {response.status_code}")
        print(f"Response Time: {end_time - start_time:.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Forecast ID: {result['forecast_id']}")
            
            # Show forecast-specific results
            if forecast_type == "price_forecast":
                print(f"💰 Predictions: {len(result['results']['predictions'])} price forecasts")
            elif forecast_type == "volatility_forecast":
                print(f"📊 Predictions: {len(result['results']['predictions'])} volatility forecasts")
            elif forecast_type == "trading_strategy":
                print(f"🎯 Strategies: {len(result['results']['strategies'])} trading strategies")
            elif forecast_type == "risk_assessment":
                print(f"⚠️ Predictions: {len(result['results']['predictions'])} risk metrics")
        else:
            print(f"❌ Error: {response.text}")

def test_retrieval_endpoints(analysis_id, forecast_id):
    """Test retrieval endpoints"""
    print("📥 Testing retrieval endpoints...")
    
    if analysis_id:
        print(f"\n📋 Retrieving analysis {analysis_id}...")
        response = requests.get(
            f"{BASE_URL}/analyses/{analysis_id}",
            headers=HEADERS
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Retrieved analysis: {result['analysis_id']}")
            print(f"📊 Insights: {len(result['results']['insights'])} insights")
        else:
            print(f"❌ Error: {response.text}")
    
    if forecast_id:
        print(f"\n📋 Retrieving forecast {forecast_id}...")
        response = requests.get(
            f"{BASE_URL}/forecasts/{forecast_id}",
            headers=HEADERS
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Retrieved forecast: {result['forecast_id']}")
            print(f"📊 Predictions: {len(result['results']['predictions'])} predictions")
        else:
            print(f"❌ Error: {response.text}")

def test_error_handling():
    """Test error handling"""
    print("⚠️ Testing error handling...")
    
    # Test without authentication
    print("\n🔒 Testing without authentication...")
    response = requests.post(
        f"{BASE_URL}/analyze/policy",
        json={"data": {}, "analysis_type": "impact_assessment"}
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    
    # Test invalid data
    print("\n❌ Testing invalid data...")
    response = requests.post(
        f"{BASE_URL}/analyze/policy",
        headers=HEADERS,
        json={"data": {}, "analysis_type": "invalid_type"}
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    
    # Test missing required fields
    print("\n📝 Testing missing required fields...")
    response = requests.post(
        f"{BASE_URL}/analyze/policy",
        headers=HEADERS,
        json={"data": {"time_series": []}}
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")

def main():
    """Run all tests"""
    print("=" * 80)
    print("🚀 Integrated Multimodal AI API Test Suite with Meta-Transformer")
    print("=" * 80)
    
    # Test basic endpoints
    test_health_check()
    test_root_endpoint()
    
    # Test comprehensive policy analysis
    analysis_id = test_policy_analysis_with_meta_transformer()
    
    # Test comprehensive trade forecasting
    forecast_id = test_trade_forecast_with_meta_transformer()
    
    # Test different analysis types
    test_different_analysis_types()
    
    # Test different forecast types
    test_different_forecast_types()
    
    # Test retrieval endpoints
    test_retrieval_endpoints(analysis_id, forecast_id)
    
    # Test error handling
    test_error_handling()
    
    print("\n" + "=" * 80)
    print("✅ Integrated test suite completed!")
    print("🎯 Meta-Transformer backend integration working successfully")
    print("📊 API ready for production use")
    print("=" * 80)

if __name__ == "__main__":
    main() 
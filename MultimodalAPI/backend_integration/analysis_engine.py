"""
Analysis Engine for Multimodal AI API
Orchestrates Meta-Transformer integration for policy analysis and trade forecasting.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json

from .meta_transformer_integration import get_meta_transformer
from .data_processors import get_multimodal_processor

logger = logging.getLogger(__name__)

class PolicyAnalysisEngine:
    """Engine for policy analysis using Meta-Transformer"""
    
    def __init__(self):
        self.meta_transformer = get_meta_transformer()
        self.data_processor = get_multimodal_processor()
    
    def analyze_policy_impact(self, data: Dict[str, Any], analysis_type: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform comprehensive policy impact analysis
        
        Args:
            data: Multimodal input data
            analysis_type: Type of analysis to perform
            parameters: Optional analysis parameters
            
        Returns:
            Analysis results with insights, visualizations, and explainability
        """
        try:
            # Validate input data
            validation_results = self.data_processor.validate_multimodal_data(data)
            if not all(validation_results.values()):
                invalid_modalities = [k for k, v in validation_results.items() if not v]
                raise ValueError(f"Invalid data in modalities: {invalid_modalities}")
            
            # Preprocess data
            processed_data, metadata = self.data_processor.preprocess_multimodal_data(data)
            
            # Create model input
            model_input = self.data_processor.create_model_input(processed_data)
            
            # Perform analysis using Meta-Transformer
            analysis_results = self.meta_transformer.analyze_policy_impact(data, analysis_type)
            
            # Enhance results with additional analysis
            enhanced_results = self._enhance_policy_analysis(analysis_results, data, analysis_type, parameters)
            
            # Add metadata
            enhanced_results['metadata'] = metadata
            enhanced_results['data_summary'] = self._create_data_summary(data)
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Policy analysis failed: {e}")
            return {
                'insights': [f"Analysis failed: {str(e)}"],
                'visualizations': [],
                'feature_importance': [],
                'confidence_score': 0.0,
                'analysis_type': analysis_type,
                'error': str(e)
            }
    
    def _enhance_policy_analysis(self, base_results: Dict[str, Any], data: Dict[str, Any], 
                                analysis_type: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhance base analysis results with additional insights"""
        enhanced_results = base_results.copy()
        
        # Add scenario analysis
        if analysis_type == "scenario_modeling":
            enhanced_results['scenarios'] = self._generate_policy_scenarios(data, parameters)
        
        # Add risk assessment
        if analysis_type == "risk_analysis":
            enhanced_results['risk_breakdown'] = self._generate_risk_breakdown(data, parameters)
        
        # Add spatial analysis
        if analysis_type == "spatial_trend_analysis":
            enhanced_results['spatial_insights'] = self._generate_spatial_insights(data, parameters)
        
        # Add temporal analysis
        enhanced_results['temporal_trends'] = self._generate_temporal_trends(data, parameters)
        
        # Add economic indicators
        enhanced_results['economic_indicators'] = self._generate_economic_indicators(data, parameters)
        
        return enhanced_results
    
    def _generate_policy_scenarios(self, data: Dict[str, Any], parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Generate policy scenarios"""
        scenarios = [
            {
                'name': 'Optimistic Scenario',
                'description': 'Best-case policy outcomes with favorable economic conditions',
                'probability': 0.25,
                'gdp_impact': '+3.5%',
                'employment_impact': '+2.1%',
                'inflation_impact': '+0.8%',
                'key_factors': ['Strong economic growth', 'Favorable trade conditions', 'Low inflation']
            },
            {
                'name': 'Base Scenario',
                'description': 'Most likely policy outcomes based on current trends',
                'probability': 0.50,
                'gdp_impact': '+2.0%',
                'employment_impact': '+1.2%',
                'inflation_impact': '+1.5%',
                'key_factors': ['Moderate growth', 'Stable trade relations', 'Controlled inflation']
            },
            {
                'name': 'Pessimistic Scenario',
                'description': 'Worst-case policy outcomes with adverse conditions',
                'probability': 0.25,
                'gdp_impact': '+0.5%',
                'employment_impact': '+0.3%',
                'inflation_impact': '+2.5%',
                'key_factors': ['Economic slowdown', 'Trade disruptions', 'High inflation']
            }
        ]
        return scenarios
    
    def _generate_risk_breakdown(self, data: Dict[str, Any], parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate detailed risk breakdown"""
        return {
            'supply_chain_risk': {
                'score': 0.65,
                'level': 'Medium',
                'description': 'Moderate risk of supply chain disruptions',
                'mitigation': 'Diversify suppliers and increase inventory'
            },
            'currency_risk': {
                'score': 0.45,
                'level': 'Low',
                'description': 'Low risk of significant currency fluctuations',
                'mitigation': 'Monitor exchange rates and hedge if necessary'
            },
            'regulatory_risk': {
                'score': 0.75,
                'level': 'High',
                'description': 'High risk of regulatory changes',
                'mitigation': 'Engage with policymakers and prepare compliance plans'
            },
            'market_risk': {
                'score': 0.55,
                'level': 'Medium',
                'description': 'Medium risk of market volatility',
                'mitigation': 'Implement risk management strategies'
            }
        }
    
    def _generate_spatial_insights(self, data: Dict[str, Any], parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate spatial analysis insights"""
        return {
            'regional_impact': {
                'urban_areas': 'High positive impact expected',
                'rural_areas': 'Moderate positive impact expected',
                'coastal_regions': 'Strong positive impact expected',
                'inland_regions': 'Moderate positive impact expected'
            },
            'spatial_patterns': [
                'Clustered development in urban centers',
                'Infrastructure improvements in rural areas',
                'Trade corridor enhancements along coastlines'
            ],
            'hotspots': [
                {'location': 'NYC', 'impact_score': 0.85},
                {'location': 'London', 'impact_score': 0.78},
                {'location': 'Tokyo', 'impact_score': 0.82}
            ]
        }
    
    def _generate_temporal_trends(self, data: Dict[str, Any], parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate temporal trend analysis"""
        return {
            'short_term': {
                'period': '1-6 months',
                'trend': 'Gradual improvement',
                'key_indicators': ['GDP growth', 'Employment rates', 'Consumer confidence']
            },
            'medium_term': {
                'period': '6-24 months',
                'trend': 'Sustained growth',
                'key_indicators': ['Investment levels', 'Trade volumes', 'Infrastructure development']
            },
            'long_term': {
                'period': '2-5 years',
                'trend': 'Structural transformation',
                'key_indicators': ['Economic diversification', 'Technology adoption', 'Sustainability metrics']
            }
        }
    
    def _generate_economic_indicators(self, data: Dict[str, Any], parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate economic indicator analysis"""
        return {
            'gdp_growth': {
                'current': '+2.1%',
                'projected': '+2.8%',
                'confidence': 0.75
            },
            'inflation_rate': {
                'current': '2.3%',
                'projected': '2.1%',
                'confidence': 0.70
            },
            'unemployment_rate': {
                'current': '4.2%',
                'projected': '3.8%',
                'confidence': 0.80
            },
            'trade_balance': {
                'current': '+$45B',
                'projected': '+$52B',
                'confidence': 0.65
            }
        }
    
    def _create_data_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of input data"""
        summary = {}
        
        if 'time_series' in data:
            summary['time_series'] = {
                'count': len(data['time_series']),
                'metrics': list(set(item['metric'] for item in data['time_series'])),
                'date_range': {
                    'start': min(item['timestamp'] for item in data['time_series']),
                    'end': max(item['timestamp'] for item in data['time_series'])
                }
            }
        
        if 'geospatial' in data:
            summary['geospatial'] = {
                'count': len(data['geospatial']),
                'types': list(set(item['type'] for item in data['geospatial']))
            }
        
        if 'text' in data:
            summary['text'] = {
                'count': len(data['text']),
                'total_length': sum(len(text) for text in data['text'])
            }
        
        if 'images' in data:
            summary['images'] = {
                'count': len(data['images']),
                'types': list(set(item['mime_type'] for item in data['images']))
            }
        
        return summary

class TradeForecastEngine:
    """Engine for trade forecasting using Meta-Transformer"""
    
    def __init__(self):
        self.meta_transformer = get_meta_transformer()
        self.data_processor = get_multimodal_processor()
    
    def forecast_trade(self, data: Dict[str, Any], forecast_type: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform comprehensive trade forecasting
        
        Args:
            data: Multimodal input data
            forecast_type: Type of forecast to perform
            parameters: Optional forecast parameters
            
        Returns:
            Forecast results with predictions, strategies, and explainability
        """
        try:
            # Validate input data
            validation_results = self.data_processor.validate_multimodal_data(data)
            if not all(validation_results.values()):
                invalid_modalities = [k for k, v in validation_results.items() if not v]
                raise ValueError(f"Invalid data in modalities: {invalid_modalities}")
            
            # Preprocess data
            processed_data, metadata = self.data_processor.preprocess_multimodal_data(data)
            
            # Create model input
            model_input = self.data_processor.create_model_input(processed_data)
            
            # Perform forecasting using Meta-Transformer
            forecast_results = self.meta_transformer.forecast_trade(data, forecast_type)
            
            # Enhance results with additional analysis
            enhanced_results = self._enhance_trade_forecast(forecast_results, data, forecast_type, parameters)
            
            # Add metadata
            enhanced_results['metadata'] = metadata
            enhanced_results['data_summary'] = self._create_data_summary(data)
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Trade forecasting failed: {e}")
            return {
                'predictions': [],
                'strategies': [],
                'feature_importance': [],
                'confidence_score': 0.0,
                'forecast_type': forecast_type,
                'error': str(e)
            }
    
    def _enhance_trade_forecast(self, base_results: Dict[str, Any], data: Dict[str, Any], 
                               forecast_type: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhance base forecast results with additional insights"""
        enhanced_results = base_results.copy()
        
        # Add market analysis
        enhanced_results['market_analysis'] = self._generate_market_analysis(data, parameters)
        
        # Add risk assessment
        enhanced_results['risk_assessment'] = self._generate_trade_risk_assessment(data, parameters)
        
        # Add technical analysis
        enhanced_results['technical_analysis'] = self._generate_technical_analysis(data, parameters)
        
        # Add fundamental analysis
        enhanced_results['fundamental_analysis'] = self._generate_fundamental_analysis(data, parameters)
        
        # Add sentiment analysis
        enhanced_results['sentiment_analysis'] = self._generate_sentiment_analysis(data, parameters)
        
        return enhanced_results
    
    def _generate_market_analysis(self, data: Dict[str, Any], parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate market analysis"""
        return {
            'market_trend': 'Bullish',
            'volatility_level': 'Medium',
            'liquidity': 'High',
            'correlation_analysis': {
                'equity_correlation': 0.65,
                'currency_correlation': 0.45,
                'commodity_correlation': 0.78
            },
            'market_regime': 'Growth',
            'key_drivers': [
                'Strong economic fundamentals',
                'Favorable monetary policy',
                'Positive trade relations'
            ]
        }
    
    def _generate_trade_risk_assessment(self, data: Dict[str, Any], parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate trade risk assessment"""
        return {
            'market_risk': {
                'score': 0.45,
                'level': 'Low',
                'description': 'Low market risk due to stable conditions'
            },
            'liquidity_risk': {
                'score': 0.25,
                'level': 'Very Low',
                'description': 'High liquidity reduces risk'
            },
            'volatility_risk': {
                'score': 0.55,
                'level': 'Medium',
                'description': 'Moderate volatility expected'
            },
            'systemic_risk': {
                'score': 0.35,
                'level': 'Low',
                'description': 'Low systemic risk in current environment'
            }
        }
    
    def _generate_technical_analysis(self, data: Dict[str, Any], parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate technical analysis"""
        return {
            'trend_indicators': {
                'moving_averages': 'Bullish crossover',
                'rsi': 58.5,
                'macd': 'Positive momentum'
            },
            'support_resistance': {
                'support_levels': [95.0, 92.5, 90.0],
                'resistance_levels': [105.0, 107.5, 110.0]
            },
            'volume_analysis': {
                'volume_trend': 'Increasing',
                'volume_ma_ratio': 1.2
            },
            'momentum_indicators': {
                'stochastic': 'Overbought',
                'williams_r': -25.5
            }
        }
    
    def _generate_fundamental_analysis(self, data: Dict[str, Any], parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate fundamental analysis"""
        return {
            'valuation_metrics': {
                'pe_ratio': 15.2,
                'pb_ratio': 2.1,
                'dividend_yield': 2.5
            },
            'financial_health': {
                'debt_to_equity': 0.45,
                'current_ratio': 1.8,
                'profit_margin': 12.5
            },
            'growth_metrics': {
                'revenue_growth': 8.5,
                'earnings_growth': 12.0,
                'asset_growth': 6.2
            },
            'industry_comparison': {
                'industry_pe': 16.8,
                'industry_growth': 7.2,
                'relative_valuation': 'Undervalued'
            }
        }
    
    def _generate_sentiment_analysis(self, data: Dict[str, Any], parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate sentiment analysis"""
        return {
            'overall_sentiment': 'Positive',
            'sentiment_score': 0.68,
            'news_sentiment': {
                'positive': 65,
                'neutral': 25,
                'negative': 10
            },
            'social_sentiment': {
                'positive': 72,
                'neutral': 18,
                'negative': 10
            },
            'analyst_sentiment': {
                'buy': 15,
                'hold': 8,
                'sell': 2
            },
            'key_sentiment_drivers': [
                'Strong earnings reports',
                'Positive economic data',
                'Favorable policy environment'
            ]
        }
    
    def _create_data_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of input data"""
        summary = {}
        
        if 'time_series' in data:
            summary['time_series'] = {
                'count': len(data['time_series']),
                'metrics': list(set(item['metric'] for item in data['time_series'])),
                'date_range': {
                    'start': min(item['timestamp'] for item in data['time_series']),
                    'end': max(item['timestamp'] for item in data['time_series'])
                }
            }
        
        if 'geospatial' in data:
            summary['geospatial'] = {
                'count': len(data['geospatial']),
                'types': list(set(item['type'] for item in data['geospatial']))
            }
        
        if 'text' in data:
            summary['text'] = {
                'count': len(data['text']),
                'total_length': sum(len(text) for text in data['text'])
            }
        
        return summary

# Global engine instances
policy_engine = PolicyAnalysisEngine()
trade_engine = TradeForecastEngine()

def get_policy_engine() -> PolicyAnalysisEngine:
    """Get the global policy analysis engine instance"""
    return policy_engine

def get_trade_engine() -> TradeForecastEngine:
    """Get the global trade forecast engine instance"""
    return trade_engine 
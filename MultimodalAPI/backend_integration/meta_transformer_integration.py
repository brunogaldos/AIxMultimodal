"""
Meta-Transformer Integration for Multimodal AI API
Integrates the Meta-Transformer foundation model with FastAPI for policy analysis and trade decisions.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import sys
import os

# Add backend_source to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../backend_source'))

from Data2Seq.Data2Seq import Data2Seq
from Data2Seq.Time_Series import DataEmbedding
from Time_Series.models.MetaTransformer import Model as MetaTransformerModel
from Time_Series.layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from Time_Series.layers.SelfAttention_Family import FullAttention, AttentionLayer
from Time_Series.layers.Embed import DataEmbedding as TimeSeriesEmbedding

logger = logging.getLogger(__name__)

class MetaTransformerIntegration:
    """
    Integration class for Meta-Transformer foundation model
    Handles multimodal data processing and analysis
    """
    
    def __init__(self, model_path: str = None, device: str = "cpu"):
        """
        Initialize Meta-Transformer integration
        
        Args:
            model_path: Path to pretrained Meta-Transformer weights
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        
        # Initialize tokenizers for different modalities
        self.tokenizers = {
            'time-series': Data2Seq(modality='time-series', dim=768),
            'text': Data2Seq(modality='text', dim=768),
            'image': Data2Seq(modality='image', dim=768),
            'video': Data2Seq(modality='video', dim=768),
            'audio': Data2Seq(modality='audio', dim=768),
            'graph': Data2Seq(modality='graph', dim=768),
            'hyper': Data2Seq(modality='hyper', dim=768),
        }
        
        # Initialize Meta-Transformer encoder
        self.encoder = self._initialize_encoder()
        
        # Task-specific heads
        self.policy_analysis_head = self._initialize_policy_head()
        self.trade_forecast_head = self._initialize_trade_head()
        
        logger.info(f"Meta-Transformer integration initialized on {self.device}")
    
    def _initialize_encoder(self) -> nn.Module:
        """Initialize the Meta-Transformer encoder"""
        try:
            # Create encoder with base configuration
            encoder = nn.Sequential(*[
                nn.TransformerEncoderLayer(
                    d_model=768,
                    nhead=12,
                    dim_feedforward=3072,
                    dropout=0.1,
                    activation='gelu',
                    batch_first=True
                )
                for i in range(12)
            ])
            
            # Load pretrained weights if available
            if self.model_path and os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location=self.device)
                encoder.load_state_dict(checkpoint, strict=False)
                logger.info(f"Loaded pretrained weights from {self.model_path}")
            else:
                logger.warning("No pretrained weights found, using random initialization")
            
            encoder.to(self.device)
            return encoder
            
        except Exception as e:
            logger.error(f"Error initializing encoder: {e}")
            # Fallback to simple transformer
            return nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=768, nhead=12, batch_first=True),
                num_layers=6
            ).to(self.device)
    
    def _initialize_policy_head(self) -> nn.Module:
        """Initialize policy analysis head"""
        return nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),  # Policy insights embedding
        ).to(self.device)
    
    def _initialize_trade_head(self) -> nn.Module:
        """Initialize trade forecasting head"""
        return nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),  # Trade predictions embedding
        ).to(self.device)
    
    def tokenize_multimodal_data(self, data: Dict[str, Any]) -> torch.Tensor:
        """
        Tokenize multimodal data using Meta-Transformer tokenizers
        
        Args:
            data: Dictionary containing different modalities
            
        Returns:
            Tokenized features tensor
        """
        features = []
        
        # Process time-series data
        if 'time_series' in data and data['time_series']:
            time_series_data = self._prepare_time_series(data['time_series'])
            if time_series_data is not None:
                time_features = self.tokenizers['time-series'](time_series_data)
                features.append(time_features)
        
        # Process text data
        if 'text' in data and data['text']:
            text_data = self._prepare_text(data['text'])
            if text_data is not None:
                text_features = self.tokenizers['text'](text_data)
                features.append(text_features)
        
        # Process image data
        if 'images' in data and data['images']:
            image_data = self._prepare_images(data['images'])
            if image_data is not None:
                image_features = self.tokenizers['image'](image_data)
                features.append(image_features)
        
        # Process geospatial data (convert to graph representation)
        if 'geospatial' in data and data['geospatial']:
            graph_data = self._prepare_geospatial(data['geospatial'])
            if graph_data is not None:
                graph_features = self.tokenizers['graph'](graph_data)
                features.append(graph_features)
        
        if not features:
            raise ValueError("No valid multimodal data provided")
        
        # Concatenate features
        combined_features = torch.cat(features, dim=1)
        return combined_features
    
    def _prepare_time_series(self, time_series_data: List[Dict]) -> Optional[torch.Tensor]:
        """Prepare time-series data for tokenization"""
        try:
            # Convert to DataFrame for easier processing
            df = pd.DataFrame(time_series_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Normalize values
            values = df['value'].values.reshape(-1, 1)
            values = (values - values.mean()) / (values.std() + 1e-8)
            
            # Convert to tensor
            tensor_data = torch.FloatTensor(values).unsqueeze(0)  # Add batch dimension
            return tensor_data.to(self.device)
            
        except Exception as e:
            logger.error(f"Error preparing time-series data: {e}")
            return None
    
    def _prepare_text(self, text_data: List[str]) -> Optional[str]:
        """Prepare text data for tokenization"""
        try:
            # Combine all text
            combined_text = " ".join(text_data)
            return combined_text
            
        except Exception as e:
            logger.error(f"Error preparing text data: {e}")
            return None
    
    def _prepare_images(self, image_data: List[Dict]) -> Optional[torch.Tensor]:
        """Prepare image data for tokenization"""
        try:
            # For now, return None as image processing requires additional setup
            # In production, this would load and preprocess images
            logger.info("Image processing not implemented in this version")
            return None
            
        except Exception as e:
            logger.error(f"Error preparing image data: {e}")
            return None
    
    def _prepare_geospatial(self, geospatial_data: List[Dict]) -> Optional[Dict]:
        """Prepare geospatial data as graph representation"""
        try:
            # Convert geospatial data to graph format
            nodes = []
            edges = []
            
            for i, point in enumerate(geospatial_data):
                nodes.append({
                    'id': i,
                    'coordinates': point['coordinates'],
                    'properties': point.get('properties', {})
                })
                
                # Create edges between nearby points
                for j, other_point in enumerate(geospatial_data[i+1:], i+1):
                    # Simple distance-based edge creation
                    dist = np.sqrt(sum((np.array(point['coordinates']) - np.array(other_point['coordinates']))**2))
                    if dist < 0.1:  # Threshold for edge creation
                        edges.append([i, j])
            
            return {
                'nodes': nodes,
                'edges': edges
            }
            
        except Exception as e:
            logger.error(f"Error preparing geospatial data: {e}")
            return None
    
    def analyze_policy_impact(self, data: Dict[str, Any], analysis_type: str) -> Dict[str, Any]:
        """
        Perform policy impact analysis using Meta-Transformer
        
        Args:
            data: Multimodal input data
            analysis_type: Type of analysis to perform
            
        Returns:
            Analysis results with insights and predictions
        """
        try:
            # Tokenize multimodal data
            features = self.tokenize_multimodal_data(data)
            
            # Encode features using Meta-Transformer
            with torch.no_grad():
                encoded_features = self.encoder(features)
                
                # Apply policy analysis head
                policy_features = self.policy_analysis_head(encoded_features.mean(dim=1))
                
                # Generate insights based on analysis type
                insights = self._generate_policy_insights(policy_features, analysis_type, data)
                
                # Calculate feature importance
                feature_importance = self._calculate_feature_importance(features, encoded_features)
                
                # Generate visualizations
                visualizations = self._generate_visualizations(analysis_type, encoded_features)
                
                return {
                    'insights': insights,
                    'visualizations': visualizations,
                    'feature_importance': feature_importance,
                    'confidence_score': float(torch.sigmoid(policy_features.mean()).item()),
                    'analysis_type': analysis_type
                }
                
        except Exception as e:
            logger.error(f"Error in policy analysis: {e}")
            return {
                'insights': [f"Analysis failed: {str(e)}"],
                'visualizations': [],
                'feature_importance': [],
                'confidence_score': 0.0,
                'analysis_type': analysis_type
            }
    
    def forecast_trade(self, data: Dict[str, Any], forecast_type: str) -> Dict[str, Any]:
        """
        Perform trade forecasting using Meta-Transformer
        
        Args:
            data: Multimodal input data
            forecast_type: Type of forecast to perform
            
        Returns:
            Forecast results with predictions and strategies
        """
        try:
            # Tokenize multimodal data
            features = self.tokenize_multimodal_data(data)
            
            # Encode features using Meta-Transformer
            with torch.no_grad():
                encoded_features = self.encoder(features)
                
                # Apply trade forecasting head
                trade_features = self.trade_forecast_head(encoded_features.mean(dim=1))
                
                # Generate predictions based on forecast type
                predictions = self._generate_trade_predictions(trade_features, forecast_type, data)
                
                # Generate trading strategies
                strategies = self._generate_trading_strategies(trade_features, forecast_type)
                
                # Calculate feature importance
                feature_importance = self._calculate_feature_importance(features, encoded_features)
                
                return {
                    'predictions': predictions,
                    'strategies': strategies,
                    'feature_importance': feature_importance,
                    'confidence_score': float(torch.sigmoid(trade_features.mean()).item()),
                    'forecast_type': forecast_type
                }
                
        except Exception as e:
            logger.error(f"Error in trade forecasting: {e}")
            return {
                'predictions': [],
                'strategies': [],
                'feature_importance': [],
                'confidence_score': 0.0,
                'forecast_type': forecast_type
            }
    
    def _generate_policy_insights(self, policy_features: torch.Tensor, analysis_type: str, data: Dict) -> List[str]:
        """Generate policy insights based on encoded features"""
        insights = []
        
        # Convert features to probabilities
        probs = torch.sigmoid(policy_features).cpu().numpy()
        
        if analysis_type == "impact_assessment":
            impact_score = probs[0, 0] if probs.shape[1] > 0 else 0.5
            insights.extend([
                f"Policy impact assessment score: {impact_score:.2f}",
                f"Expected GDP impact: {impact_score * 5:.1f}% over 5 years",
                f"Market volatility change: {impact_score * 0.3:.2f}%"
            ])
        elif analysis_type == "scenario_modeling":
            scenario_scores = probs[0, :3] if probs.shape[1] >= 3 else [0.5, 0.3, 0.2]
            insights.extend([
                f"Best case scenario probability: {scenario_scores[0]:.2f}",
                f"Base case scenario probability: {scenario_scores[1]:.2f}",
                f"Worst case scenario probability: {scenario_scores[2]:.2f}"
            ])
        elif analysis_type == "risk_analysis":
            risk_score = probs[0, 0] if probs.shape[1] > 0 else 0.5
            insights.extend([
                f"Overall risk score: {risk_score:.2f}",
                f"Supply chain risk: {risk_score * 0.8:.2f}",
                f"Currency fluctuation risk: {risk_score * 0.6:.2f}"
            ])
        elif analysis_type == "spatial_trend_analysis":
            spatial_score = probs[0, 0] if probs.shape[1] > 0 else 0.5
            insights.extend([
                f"Spatial trend strength: {spatial_score:.2f}",
                f"Urban-rural divide impact: {spatial_score * 0.7:.2f}",
                f"Regional clustering: {spatial_score * 0.9:.2f}"
            ])
        
        return insights
    
    def _generate_trade_predictions(self, trade_features: torch.Tensor, forecast_type: str, data: Dict) -> List[Dict]:
        """Generate trade predictions based on encoded features"""
        predictions = []
        
        # Convert features to predictions
        probs = torch.sigmoid(trade_features).cpu().numpy()
        
        if forecast_type == "price_forecast":
            base_price = 100.0
            for i in range(5):
                price_change = probs[0, i] if i < probs.shape[1] else 0.02
                predictions.append({
                    "timestamp": (datetime.now() + timedelta(days=30*(i+1))).isoformat(),
                    "value": base_price * (1 + price_change * (i+1)),
                    "metric": "commodity_price"
                })
        elif forecast_type == "volatility_forecast":
            base_volatility = 0.15
            for i in range(5):
                vol_change = probs[0, i] if i < probs.shape[1] else 0.1
                predictions.append({
                    "timestamp": (datetime.now() + timedelta(days=30*(i+1))).isoformat(),
                    "value": base_volatility * (1 + vol_change),
                    "metric": "volatility"
                })
        
        return predictions
    
    def _generate_trading_strategies(self, trade_features: torch.Tensor, forecast_type: str) -> List[Dict]:
        """Generate trading strategies based on encoded features"""
        strategies = []
        
        # Convert features to strategy decisions
        probs = torch.sigmoid(trade_features).cpu().numpy()
        
        assets = ["WTI Crude Oil", "Gold", "EUR/USD", "S&P 500"]
        actions = ["buy", "sell", "hold"]
        
        for i, asset in enumerate(assets[:3]):
            if i < probs.shape[1]:
                confidence = probs[0, i]
                action_idx = int(confidence * 3) % 3
            else:
                confidence = 0.5
                action_idx = 0
            
            strategies.append({
                "action": actions[action_idx],
                "asset": asset,
                "confidence": float(confidence)
            })
        
        return strategies
    
    def _calculate_feature_importance(self, input_features: torch.Tensor, encoded_features: torch.Tensor) -> List[Dict]:
        """Calculate feature importance using attention weights"""
        try:
            # Simple feature importance based on variance
            importance_scores = torch.var(encoded_features, dim=1).cpu().numpy()
            
            feature_names = ["time_series", "text", "geospatial", "images"]
            feature_importance = []
            
            for i, (name, score) in enumerate(zip(feature_names, importance_scores[:len(feature_names)])):
                feature_importance.append({
                    "feature": name,
                    "importance": float(score)
                })
            
            return feature_importance
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            return []
    
    def _generate_visualizations(self, analysis_type: str, encoded_features: torch.Tensor) -> List[str]:
        """Generate visualization URLs (mock implementation)"""
        # In production, this would generate actual visualizations
        return [
            f"https://example.com/visuals/{analysis_type}_trend.png",
            f"https://example.com/visuals/{analysis_type}_distribution.png",
            f"https://example.com/visuals/{analysis_type}_correlation.png"
        ]

# Global instance for API use
meta_transformer = None

def initialize_meta_transformer(model_path: str = None, device: str = "cpu"):
    """Initialize the global Meta-Transformer instance"""
    global meta_transformer
    if meta_transformer is None:
        meta_transformer = MetaTransformerIntegration(model_path, device)
    return meta_transformer

def get_meta_transformer() -> MetaTransformerIntegration:
    """Get the global Meta-Transformer instance"""
    if meta_transformer is None:
        initialize_meta_transformer()
    return meta_transformer 
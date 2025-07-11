"""
Backend Integration Package for Multimodal AI API
Integrates Meta-Transformer foundation model with FastAPI for policy analysis and trade decisions.
"""

from .meta_transformer_integration import MetaTransformerIntegration, initialize_meta_transformer, get_meta_transformer
from .data_processors import MultimodalDataProcessor, get_multimodal_processor
from .analysis_engine import PolicyAnalysisEngine, TradeForecastEngine, get_policy_engine, get_trade_engine

__all__ = [
    'MetaTransformerIntegration',
    'initialize_meta_transformer',
    'get_meta_transformer',
    'MultimodalDataProcessor',
    'get_multimodal_processor',
    'PolicyAnalysisEngine',
    'TradeForecastEngine',
    'get_policy_engine',
    'get_trade_engine'
] 
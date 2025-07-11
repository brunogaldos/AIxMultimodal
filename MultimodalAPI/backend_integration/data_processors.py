"""
Data Processors for Multimodal AI API
Handles data preprocessing, validation, and transformation for different modalities.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json
import requests
from PIL import Image
import io
import base64

logger = logging.getLogger(__name__)

class TimeSeriesProcessor:
    """Processor for time-series data"""
    
    @staticmethod
    def validate_time_series_data(data: List[Dict]) -> bool:
        """Validate time-series data format"""
        if not data:
            return False
        
        required_fields = ['timestamp', 'value', 'metric']
        for item in data:
            if not all(field in item for field in required_fields):
                return False
            
            # Validate timestamp format
            try:
                datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00'))
            except ValueError:
                return False
            
            # Validate value is numeric
            if not isinstance(item['value'], (int, float)):
                return False
        
        return True
    
    @staticmethod
    def preprocess_time_series(data: List[Dict]) -> Tuple[torch.Tensor, Dict]:
        """Preprocess time-series data for model input"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Extract features
            values = df['value'].values.reshape(-1, 1)
            
            # Calculate additional features
            features = []
            features.append(values)  # Raw values
            
            # Moving averages
            for window in [3, 7, 14]:
                if len(values) >= window:
                    ma = pd.Series(values.flatten()).rolling(window=window).mean().values.reshape(-1, 1)
                    ma = np.nan_to_num(ma, nan=values.mean())
                    features.append(ma)
                else:
                    features.append(np.full_like(values, values.mean()))
            
            # Volatility (rolling standard deviation)
            if len(values) >= 7:
                volatility = pd.Series(values.flatten()).rolling(window=7).std().values.reshape(-1, 1)
                volatility = np.nan_to_num(volatility, nan=values.std())
                features.append(volatility)
            else:
                features.append(np.full_like(values, values.std()))
            
            # Trend features
            if len(values) >= 2:
                trend = np.diff(values, axis=0)
                trend = np.vstack([trend[0:1], trend])  # Pad first value
                features.append(trend)
            else:
                features.append(np.zeros_like(values))
            
            # Combine features
            combined_features = np.hstack(features)
            
            # Normalize
            mean = combined_features.mean(axis=0, keepdims=True)
            std = combined_features.std(axis=0, keepdims=True)
            normalized_features = (combined_features - mean) / (std + 1e-8)
            
            # Convert to tensor
            tensor_data = torch.FloatTensor(normalized_features).unsqueeze(0)
            
            # Metadata
            metadata = {
                'original_length': len(values),
                'features_count': combined_features.shape[1],
                'normalization_stats': {
                    'mean': mean.tolist(),
                    'std': std.tolist()
                },
                'metrics': df['metric'].unique().tolist()
            }
            
            return tensor_data, metadata
            
        except Exception as e:
            logger.error(f"Error preprocessing time-series data: {e}")
            raise ValueError(f"Time-series preprocessing failed: {str(e)}")
    
    @staticmethod
    def generate_time_features(timestamps: List[str]) -> torch.Tensor:
        """Generate time-based features"""
        try:
            time_features = []
            for ts in timestamps:
                dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                features = [
                    dt.hour / 24.0,  # Hour of day
                    dt.weekday() / 7.0,  # Day of week
                    dt.day / 31.0,  # Day of month
                    dt.month / 12.0,  # Month
                    dt.year - 2020,  # Year (normalized)
                ]
                time_features.append(features)
            
            return torch.FloatTensor(time_features)
            
        except Exception as e:
            logger.error(f"Error generating time features: {e}")
            return torch.zeros(len(timestamps), 5)

class GeospatialProcessor:
    """Processor for geospatial data"""
    
    @staticmethod
    def validate_geospatial_data(data: List[Dict]) -> bool:
        """Validate geospatial data format"""
        if not data:
            return False
        
        for item in data:
            if 'type' not in item or 'coordinates' not in item:
                return False
            
            if item['type'] not in ['Point', 'Polygon', 'LineString']:
                return False
            
            if not isinstance(item['coordinates'], list):
                return False
            
            # Validate coordinate format
            if item['type'] == 'Point':
                if len(item['coordinates']) != 2:
                    return False
                if not all(isinstance(c, (int, float)) for c in item['coordinates']):
                    return False
        
        return True
    
    @staticmethod
    def preprocess_geospatial(data: List[Dict]) -> Tuple[torch.Tensor, Dict]:
        """Preprocess geospatial data for model input"""
        try:
            # Extract coordinates and properties
            coordinates = []
            properties = []
            
            for item in data:
                if item['type'] == 'Point':
                    coords = item['coordinates']
                    coordinates.append(coords)
                    
                    # Extract properties
                    props = item.get('properties', {})
                    prop_values = [
                        float(props.get('value', 0)),
                        float(props.get('density', 0)),
                        float(props.get('elevation', 0))
                    ]
                    properties.append(prop_values)
            
            if not coordinates:
                raise ValueError("No valid coordinates found")
            
            # Convert to numpy arrays
            coords_array = np.array(coordinates)
            props_array = np.array(properties)
            
            # Normalize coordinates (simple min-max normalization)
            coords_min = coords_array.min(axis=0)
            coords_max = coords_array.max(axis=0)
            normalized_coords = (coords_array - coords_min) / (coords_max - coords_min + 1e-8)
            
            # Normalize properties
            props_min = props_array.min(axis=0)
            props_max = props_array.max(axis=0)
            normalized_props = (props_array - props_min) / (props_max - props_min + 1e-8)
            
            # Combine features
            combined_features = np.hstack([normalized_coords, normalized_props])
            
            # Convert to tensor
            tensor_data = torch.FloatTensor(combined_features).unsqueeze(0)
            
            # Metadata
            metadata = {
                'point_count': len(coordinates),
                'feature_dim': combined_features.shape[1],
                'coordinate_bounds': {
                    'min': coords_min.tolist(),
                    'max': coords_max.tolist()
                },
                'property_bounds': {
                    'min': props_min.tolist(),
                    'max': props_max.tolist()
                }
            }
            
            return tensor_data, metadata
            
        except Exception as e:
            logger.error(f"Error preprocessing geospatial data: {e}")
            raise ValueError(f"Geospatial preprocessing failed: {str(e)}")

class TextProcessor:
    """Processor for text data"""
    
    @staticmethod
    def validate_text_data(data: List[str]) -> bool:
        """Validate text data format"""
        if not data:
            return False
        
        for text in data:
            if not isinstance(text, str):
                return False
            if len(text.strip()) == 0:
                return False
        
        return True
    
    @staticmethod
    def preprocess_text(data: List[str]) -> Tuple[str, Dict]:
        """Preprocess text data for model input"""
        try:
            # Combine all text
            combined_text = " ".join(data)
            
            # Basic cleaning
            cleaned_text = combined_text.strip()
            
            # Metadata
            metadata = {
                'text_count': len(data),
                'total_length': len(cleaned_text),
                'avg_length': len(cleaned_text) / len(data) if data else 0,
                'word_count': len(cleaned_text.split())
            }
            
            return cleaned_text, metadata
            
        except Exception as e:
            logger.error(f"Error preprocessing text data: {e}")
            raise ValueError(f"Text preprocessing failed: {str(e)}")

class ImageProcessor:
    """Processor for image data"""
    
    @staticmethod
    def validate_image_data(data: List[Dict]) -> bool:
        """Validate image data format"""
        if not data:
            return False
        
        for item in data:
            if 'url' not in item or 'mime_type' not in item:
                return False
            
            # Validate MIME type
            valid_types = ['image/jpeg', 'image/png', 'image/gif', 'image/bmp']
            if item['mime_type'] not in valid_types:
                return False
        
        return True
    
    @staticmethod
    def preprocess_images(data: List[Dict]) -> Tuple[torch.Tensor, Dict]:
        """Preprocess image data for model input"""
        try:
            # For now, return placeholder tensor
            # In production, this would load and preprocess actual images
            batch_size = len(data)
            placeholder_tensor = torch.randn(batch_size, 3, 224, 224)  # Standard image size
            
            metadata = {
                'image_count': batch_size,
                'processed': False,
                'note': 'Image processing not implemented in this version'
            }
            
            return placeholder_tensor, metadata
            
        except Exception as e:
            logger.error(f"Error preprocessing image data: {e}")
            raise ValueError(f"Image preprocessing failed: {str(e)}")

class MultimodalDataProcessor:
    """Main processor for handling multimodal data"""
    
    def __init__(self):
        self.time_series_processor = TimeSeriesProcessor()
        self.geospatial_processor = GeospatialProcessor()
        self.text_processor = TextProcessor()
        self.image_processor = ImageProcessor()
    
    def validate_multimodal_data(self, data: Dict[str, Any]) -> Dict[str, bool]:
        """Validate all modalities in the data"""
        validation_results = {}
        
        # Validate time-series data
        if 'time_series' in data:
            validation_results['time_series'] = self.time_series_processor.validate_time_series_data(data['time_series'])
        else:
            validation_results['time_series'] = True  # Optional modality
        
        # Validate geospatial data
        if 'geospatial' in data:
            validation_results['geospatial'] = self.geospatial_processor.validate_geospatial_data(data['geospatial'])
        else:
            validation_results['geospatial'] = True  # Optional modality
        
        # Validate text data
        if 'text' in data:
            validation_results['text'] = self.text_processor.validate_text_data(data['text'])
        else:
            validation_results['text'] = True  # Optional modality
        
        # Validate image data
        if 'images' in data:
            validation_results['images'] = self.image_processor.validate_image_data(data['images'])
        else:
            validation_results['images'] = True  # Optional modality
        
        return validation_results
    
    def preprocess_multimodal_data(self, data: Dict[str, Any]) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Preprocess all modalities in the data"""
        processed_data = {}
        metadata = {}
        
        # Process time-series data
        if 'time_series' in data and data['time_series']:
            try:
                tensor_data, ts_metadata = self.time_series_processor.preprocess_time_series(data['time_series'])
                processed_data['time_series'] = tensor_data
                metadata['time_series'] = ts_metadata
            except Exception as e:
                logger.error(f"Time-series processing failed: {e}")
                raise
        
        # Process geospatial data
        if 'geospatial' in data and data['geospatial']:
            try:
                tensor_data, geo_metadata = self.geospatial_processor.preprocess_geospatial(data['geospatial'])
                processed_data['geospatial'] = tensor_data
                metadata['geospatial'] = geo_metadata
            except Exception as e:
                logger.error(f"Geospatial processing failed: {e}")
                raise
        
        # Process text data
        if 'text' in data and data['text']:
            try:
                text_data, text_metadata = self.text_processor.preprocess_text(data['text'])
                processed_data['text'] = text_data
                metadata['text'] = text_metadata
            except Exception as e:
                logger.error(f"Text processing failed: {e}")
                raise
        
        # Process image data
        if 'images' in data and data['images']:
            try:
                tensor_data, img_metadata = self.image_processor.preprocess_images(data['images'])
                processed_data['images'] = tensor_data
                metadata['images'] = img_metadata
            except Exception as e:
                logger.error(f"Image processing failed: {e}")
                raise
        
        if not processed_data:
            raise ValueError("No valid data to process")
        
        return processed_data, metadata
    
    def create_model_input(self, processed_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Create unified model input from processed data"""
        try:
            # Combine different modalities
            features = []
            
            if 'time_series' in processed_data:
                # Flatten time-series features
                ts_features = processed_data['time_series'].flatten(1)
                features.append(ts_features)
            
            if 'geospatial' in processed_data:
                # Flatten geospatial features
                geo_features = processed_data['geospatial'].flatten(1)
                features.append(geo_features)
            
            if 'images' in processed_data:
                # Flatten image features
                img_features = processed_data['images'].flatten(1)
                features.append(img_features)
            
            # Combine all features
            if features:
                combined_features = torch.cat(features, dim=1)
                
                # Pad or truncate to fixed size if needed
                target_size = 768  # Meta-Transformer input size
                if combined_features.shape[1] > target_size:
                    combined_features = combined_features[:, :target_size]
                elif combined_features.shape[1] < target_size:
                    padding = torch.zeros(combined_features.shape[0], target_size - combined_features.shape[1])
                    combined_features = torch.cat([combined_features, padding], dim=1)
                
                return combined_features
            else:
                raise ValueError("No features to combine")
                
        except Exception as e:
            logger.error(f"Error creating model input: {e}")
            raise ValueError(f"Model input creation failed: {str(e)}")

# Global processor instance
multimodal_processor = MultimodalDataProcessor()

def get_multimodal_processor() -> MultimodalDataProcessor:
    """Get the global multimodal processor instance"""
    return multimodal_processor 
"""Configuration management utilities."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages configuration loading and merging for experiments."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.base_config = None
        
    def load_base_config(self) -> Dict[str, Any]:
        """Load the base configuration file."""
        base_config_path = self.config_dir / "base_config.yaml"
        
        if not base_config_path.exists():
            raise FileNotFoundError(f"Base config not found: {base_config_path}")
            
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
            
        logger.info(f"Loaded base configuration from {base_config_path}")
        return self.base_config
    
    def load_dataset_config(self, dataset_type: str) -> Dict[str, Any]:
        """Load dataset-specific configuration."""
        config_path = self.config_dir / f"{dataset_type}_config.yaml"
        
        if not config_path.exists():
            logger.warning(f"Dataset config not found: {config_path}")
            return {}
            
        with open(config_path, 'r') as f:
            dataset_config = yaml.safe_load(f)
            
        logger.info(f"Loaded {dataset_type} configuration from {config_path}")
        return dataset_config
    
    def load_experiment_config(self, experiment_name: str) -> Dict[str, Any]:
        """Load experiment-specific configuration."""
        config_path = self.config_dir / "experiment_configs" / f"{experiment_name}.yaml"
        
        if not config_path.exists():
            logger.warning(f"Experiment config not found: {config_path}")
            return {}
            
        with open(config_path, 'r') as f:
            experiment_config = yaml.safe_load(f)
            
        logger.info(f"Loaded experiment configuration from {config_path}")
        return experiment_config
    
    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple configuration dictionaries."""
        merged = {}
        
        for config in configs:
            if config:
                merged = self._deep_merge(merged, config)
                
        return merged
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
                
        return result
    
    def get_config(self, dataset_type: Optional[str] = None, 
                   experiment_name: Optional[str] = None) -> Dict[str, Any]:
        """Get merged configuration for a specific setup."""
        if self.base_config is None:
            self.load_base_config()
            
        configs = [self.base_config]
        
        if dataset_type:
            dataset_config = self.load_dataset_config(dataset_type)
            configs.append(dataset_config)
            
        if experiment_name:
            experiment_config = self.load_experiment_config(experiment_name)
            configs.append(experiment_config)
            
        return self.merge_configs(*configs)
    
    def save_config(self, config: Dict[str, Any], output_path: str) -> None:
        """Save configuration to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
            
        logger.info(f"Saved configuration to {output_path}")
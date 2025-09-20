"""
OpsLens configuration manager
"""

import yaml
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
import logging
logging.basicConfig(filename='./eval/test/opslens.log', level=logging.INFO)


class ConfigManager:
    
    _instance = None
    _config = None
    
    def __new__(cls):
        """singleton pattern"""
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """initialize config manager"""
        if self._config is None:
            self._load_config()
    
    def _load_config(self):
        """load config file"""
        try:
            config_path = Path(__file__).parent / "dataset_config.yaml"
            if not config_path.exists():
                raise FileNotFoundError(f"Config file does not exist: {config_path}")
            
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
            
            logging.info(f"[ConfigManager] Config file loaded successfully: {config_path}")
            logging.info(f"[ConfigManager] Current dataset: {self.get_current_dataset()}")
            
        except Exception as e:
            logging.debug(f"[ConfigManager] Config file load failed: {e}")
            logging.debug(traceback.format_exc())
            
    
    
    def get_current_dataset(self) -> str:
        """get current dataset name"""
        return self._config.get("current_dataset", "bank")
    
    def get_dataset_config(self, dataset_name: Optional[str] = None) -> Dict[str, Any]:
        """get config for specified dataset"""
        if dataset_name is None:
            dataset_name = self.get_current_dataset()
        
        datasets = self._config.get("datasets", {})
        if dataset_name not in datasets:
            logging.info(f"[ConfigManager] warning: dataset '{dataset_name}' does not exist, using default config")
            return datasets.get("bank", {})
        
        return datasets[dataset_name]
    
    def get_prompt_type(self, dataset_name: Optional[str] = None) -> str:
        config = self.get_dataset_config(dataset_name)
        return config.get("prompt_type", "bank")
    
    def get_tool_ecosystem_paths(self, dataset_name: Optional[str] = None) -> Dict[str, str]:
        config = self.get_dataset_config(dataset_name)
        return config.get("tool_ecosystem", {
            "metric": "./tool_ecosystem_metric",
            "log": "./tool_ecosystem_log",
            "trace": "./tool_ecosystem_trace"
        })
    
    def get_coder_memory_paths(self, dataset_name: Optional[str] = None) -> Dict[str, str]:
        config = self.get_dataset_config(dataset_name)
        return config.get("coder_memory", {
            "chroma_storage": "./memory/coder_chroma",
            "refinement_storage": "./memory/coder_refinement"
        })
    
    def get_data_paths(self, dataset_name: Optional[str] = None) -> Dict[str, str]:
        config = self.get_dataset_config(dataset_name)
        return config.get("data_paths", {
            "base_path": "./coding/dataset",
            "relative_root": "./Bank/telemetry"
        })

    def get_tool_files(self, dataset_name: Optional[str] = None) -> Dict[str, str]:
        config = self.get_dataset_config(dataset_name)
        return config.get("tool_files", {
            "metric_tools": "tools/metric_tools.py",
            "log_tools": "tools/log_tools.py",
            "trace_tools": "tools/trace_tools.py",
            "specific_tools": "bank_tool.py"
        })

    def get_relative_root_path(self, dataset_name: Optional[str] = None) -> str:
        data_paths = self.get_data_paths(dataset_name)
        return data_paths.get("relative_root", "./Bank/telemetry")
    
    def get_global_config(self) -> Dict[str, Any]:
        return self._config.get("global_config", {})
    
    def set_current_dataset(self, dataset_name: str):
        if dataset_name not in self._config.get("datasets", {}):
            raise ValueError(f"Dataset '{dataset_name}' does not exist")
        
        self._config["current_dataset"] = dataset_name
        logging.info(f"[ConfigManager] Current dataset set to: {dataset_name}")
    
    def reload_config(self):
        self._config = None
        self._load_config()
    
    def get_openai_config(self) -> Dict[str, str]:
        global_config = self.get_global_config()
        return global_config.get("openai", {})

    def get_model_clients_config(self) -> Dict[str, Any]:
        """get model clients config from global config"""
        global_config = self.get_global_config()
        return global_config.get("model_clients", {
            "default": {
                "model": "gpt-5-mini",
                "temperature": 0,
                "model_info": {
                    "vision": True,
                    "function_calling": True,
                    "json_output": False,
                    "family": "unknown"
                }
            },
            "reason": {
                "model": "gemini-2.5-pro",
                "temperature": 0,
                "model_info": {
                    "vision": True,
                    "function_calling": True,
                    "json_output": False,
                    "family": "unknown"
                }
            }
        })


# Global config manager instance
config_manager = ConfigManager()


def get_config_manager() -> ConfigManager:
    return config_manager


def get_current_dataset() -> str:
    return config_manager.get_current_dataset()


def get_prompt_type() -> str:
    return config_manager.get_prompt_type()


def get_tool_ecosystem_paths() -> Dict[str, str]:
    return config_manager.get_tool_ecosystem_paths()


def get_coder_memory_paths() -> Dict[str, str]:
    return config_manager.get_coder_memory_paths()


def get_data_paths() -> Dict[str, str]:
    return config_manager.get_data_paths()


def get_tool_files() -> Dict[str, str]:
    return config_manager.get_tool_files()


def get_tool_files_with_env_override() -> Dict[str, str]:
    return config_manager.get_tool_files()


def get_relative_root_path() -> str:
    return config_manager.get_relative_root_path()


def get_openai_config() -> Dict[str, str]:
    return config_manager.get_openai_config()


def get_model_clients_config() -> Dict[str, Any]:
    return config_manager.get_model_clients_config()


def get_prompt_type_with_env_override() -> str:
    return get_prompt_type()


if __name__ == "__main__":
    cm = get_config_manager()
    print(f"Current dataset: {cm.get_current_dataset()}")
    print(f"Prompt type: {cm.get_prompt_type()}")
    print(f"Tool Ecosystem path: {cm.get_tool_ecosystem_paths()}")
    print(f"Coder Memory path: {cm.get_coder_memory_paths()}")
    print(f"Data path: {cm.get_data_paths()}")

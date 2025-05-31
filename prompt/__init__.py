import os
import importlib

def get_prompt_module(prompt_type="bank"):
    """
    Args:
        prompt_type: [prompt_type] default is "bank"
        
    Returns:
        module: [module] the corresponding prompt template module
    """
    module_name = f"prompt.AgentPrompt_{prompt_type}"
    print("module_name",module_name)
    try:
        return importlib.import_module(module_name)
    except ImportError:
        # if the import fails, use the default bank template
        print(f"Warning: Failed to import {module_name}, using default AgentPrompt_bank")
        return importlib.import_module("prompt.AgentPrompt_bank")
import os
import importlib

def get_prompt_module(prompt_type="bank"):
    """根据指定的 prompt_type 动态导入对应的提示模板模块
    
    Args:
        prompt_type: 提示模板类型，默认为 "bank"
        
    Returns:
        module: 对应的提示模板模块
    """
    module_name = f"prompt.AgentPrompt_{prompt_type}"
    print("module_name",module_name)
    try:
        return importlib.import_module(module_name)
    except ImportError:
        # 如果导入失败，使用默认的bank模板
        print(f"警告: 无法导入 {module_name}，使用默认的 AgentPrompt_bank")
        return importlib.import_module("prompt.AgentPrompt_bank")
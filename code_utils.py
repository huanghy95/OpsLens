import json
import os
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
# Save in json as logs, executable code is in generated_functions.py

@dataclass
class CodeBlock:
    language: str
    code: str
    timestamp: str = ""

def save_code_blocks(code_blocks: List[Dict], filename: str = "gen_code_json/saved_code_blocks.json") -> None:
    """
    Save code blocks to a JSON file
    
    Args:
        code_blocks: List containing code block information
        filename: Name of the file to save
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Prepare data to save
    blocks_to_save = []
    for block in code_blocks:
        if isinstance(block, dict):
            block['timestamp'] = current_time
            blocks_to_save.append(block)
        else:
            blocks_to_save.append({
                'language': block.language,
                'code': block.code,
                'timestamp': current_time
            })
    
    # Create directory if it doesn't exist
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    # If file exists, read existing data and append
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            try:
                existing_data = json.load(f)
                blocks_to_save = existing_data + blocks_to_save
            except json.JSONDecodeError:
                pass
    
    # Save data
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(blocks_to_save, f, ensure_ascii=False, indent=2)

def load_code_blocks(filename: str = "gen_code_json/saved_code_blocks.json") -> List[CodeBlock]:
    """
    Load code blocks from a JSON file
    
    Args:
        filename: Name of the file to load
    
    Returns:
        List containing CodeBlock objects
    """
    # Create directory if it doesn't exist
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    if not os.path.exists(filename):
        return []
    
    with open(filename, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            return [CodeBlock(**block) for block in data]
        except json.JSONDecodeError:
            return []

def save_code_as_functions(code_blocks: List[CodeBlock], description: str, output_file: str = "generated_functions.py") -> None:
    """
    Convert code blocks to Python functions and save to a file
    
    Args:
        code_blocks: List of code blocks
        output_file: Output Python file name
    """
    from datetime import datetime
    
    function_template = '''
def function_{index}_{timestamp}():
    """
    Generated time: {datetime}
    Tool description: {description}
    """
{indented_code}
'''
    
    # Create directory if it doesn't exist
    directory = os.path.dirname(output_file)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    # If file exists, read existing content first
    existing_content = ""
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_content = f.read()
    
    # Prepare new content
    new_content = existing_content if existing_content else "# Automatically generated functions file\n\n"
    
    for i, block in enumerate(code_blocks, 1):
        if block.language.lower() in ['python', 'py']:
            # Generate timestamp (for function name)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Indent code
            indented_code = "\n".join("    " + line for line in block.code.split("\n"))
            
            # Generate function
            function_code = function_template.format(
                index=i,
                timestamp=timestamp,
                datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                description=description,
                indented_code=indented_code
            )
            
            new_content += function_code + "\n"
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(new_content) 
import json
import os
import re
import uuid
from datetime import datetime
from typing import List, Dict, Tuple
from dataclasses import dataclass
import ast
import traceback
# Save in json as logs, executable code is in generated_functions.py

@dataclass
class CodeBlock:
    language: str
    code: str
    timestamp: str = ""

# Sanitize helpers to remove unguarded top-level invocations in Python code

def _is_main_guard(node: ast.If) -> bool:
    # Matches: if __name__ == '__main__':
    try:
        test = node.test
        if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq):
            left = test.left
            right = test.comparators[0] if test.comparators else None
            def is_name_main(n):
                return isinstance(n, ast.Name) and n.id == "__name__"
            def is_const_main(n):
                return (isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value == "__main__")
            return (is_name_main(left) and is_const_main(right)) or (is_const_main(left) and is_name_main(right))
        return False
    except SyntaxError as e:
        print(f"Warning: Syntax error in _is_main_guard. Error: {e}")
        return False
    except Exception as e:
        return False


def _is_constant_like(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_is_constant_like(elt) for elt in node.elts)
    if isinstance(node, ast.Dict):
        return all(_is_constant_like(k) and _is_constant_like(v) for k, v in zip(node.keys, node.values))
    return False


def sanitize_python_code(code: str) -> str:
    """
    Remove unguarded top-level invocations while preserving:
    - imports, function/class definitions
    - simple constant assignments
    - guarded blocks under: if __name__ == '__main__':
    """
    try:
        tree = ast.parse(code)
        source_lines = code.splitlines()
        kept_segments: List[List[int]] = []

        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                kept_segments.append([node.lineno, getattr(node, 'end_lineno', node.lineno)])
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = node.value if hasattr(node, 'value') else None
                if value is not None and _is_constant_like(value):
                    kept_segments.append([node.lineno, getattr(node, 'end_lineno', node.lineno)])
            elif isinstance(node, ast.If) and _is_main_guard(node):
                kept_segments.append([node.lineno, getattr(node, 'end_lineno', node.lineno)])
            # else: drop unguarded executable statements (Expr/Call etc.)

        if not kept_segments:
            return ""

        kept_segments.sort(key=lambda seg: seg[0])
        merged: List[List[int]] = []
        for start, end in kept_segments:
            if not merged or start > merged[-1][1] + 1:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)

        output_lines: List[str] = []
        for start, end in merged:
            output_lines.extend(source_lines[start - 1:end])

        sanitized = "\n".join(output_lines).rstrip() + "\n"
        return sanitized
    except SyntaxError as e:
        print(f"Warning: Syntax error in code, returning original code. Error: {e}")
        return code
    except Exception as e:
        print(traceback.format_exc())
        return code


def add_uuid_to_function_names(code: str) -> Tuple[str, Dict[str, str]]:
    """
    Add UUID suffix to function names in Python code to avoid naming conflicts

    Args:
        code: Python code string

    Returns:
        Tuple[str, Dict[str, str]]: Modified code and mapping of original->new function names
    """
    if not isinstance(code, str):
        return code, {}

    try:
        tree = ast.parse(code)
        source_lines = code.splitlines()
        function_mapping = {}

        # Find all function definitions and generate UUIDs for them
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                original_name = node.name
                # Generate a short UUID (8 characters)
                short_uuid = str(uuid.uuid4()).replace('-', '')[:8]
                new_name = f"{original_name}_{short_uuid}"
                function_mapping[original_name] = new_name

        # Replace function definitions and calls
        modified_code = code
        for original_name, new_name in function_mapping.items():
            # Replace function definitions
            # Pattern for function definition: def function_name( or async def function_name(
            def_pattern = rf'\b(async\s+)?def\s+{re.escape(original_name)}\s*\('
            modified_code = re.sub(def_pattern, rf'\1def {new_name}(', modified_code)

            # Replace function calls (be careful not to replace method calls or attributes)
            # Only replace standalone function calls, not method calls (no dot before)
            call_pattern = rf'(?<!\.)(?<!\w){re.escape(original_name)}\s*\('
            modified_code = re.sub(call_pattern, f'{new_name}(', modified_code)

        return modified_code, function_mapping

    except SyntaxError as e:
        print(f"Warning: Syntax error in add_uuid_to_function_names, returning original code. Error: {e}")
        return code, {}
    except Exception as e:
        print(f"Warning: Error in add_uuid_to_function_names: {e}")
        print(traceback.format_exc())
        return code, {}


def convert_relative_to_absolute_paths(code: str, base_path: str) -> str:
    """
    Convert relative paths in code to absolute paths

    Args:
        code: Python code string
        base_path: Base path for Docker executor work directory

    Returns:
        str: Code with relative paths converted to absolute paths
    """
    if not isinstance(code, str):
        return code

    patterns = [
        (r"'(\./[^']*)'", lambda m: f"'{os.path.join(base_path, m.group(1)[2:])}'"),
        (r'"(\./[^"]*)"', lambda m: f'"{os.path.join(base_path, m.group(1)[2:])}"'),
        (r"f'([^']*\./[^']*)'", lambda m: f"f'{m.group(1).replace('./', base_path + '/')}'"),
        (r'f"([^"]*\./[^"]*)"', lambda m: f'f"{m.group(1).replace("./" , base_path + "/")}"'),
        (r'os\.path\.join\s*\(\s*["\'](\./[^"\']*)["\']',
         lambda m: f'os.path.join("{os.path.join(base_path, m.group(1)[2:])}"'),
        (r'Path\s*\(\s*["\'](\./[^"\']*)["\']',
         lambda m: f'Path("{os.path.join(base_path, m.group(1)[2:])}"'),
        (r'open\s*\(\s*["\'](\./[^"\']*)["\']',
         lambda m: f'open("{os.path.join(base_path, m.group(1)[2:])}"'),
        (r'(pd\.read_\w+)\s*\(\s*["\'](\./[^"\']*)["\']',
         lambda m: f'{m.group(1)}("{os.path.join(base_path, m.group(2)[2:])}"'),
        (r'(load_data|read_file|load_file)\s*\(\s*["\'](\./[^"\']*)["\']',
         lambda m: f'{m.group(1)}("{os.path.join(base_path, m.group(2)[2:])}"'),
    ]

    modified_code = code
    replacements_made = 0

    for pattern, replacement_func in patterns:
        def replace_match(match):
            nonlocal replacements_made
            replacements_made += 1
            return replacement_func(match)

        modified_code = re.sub(pattern, replace_match, modified_code)

    if replacements_made > 0:
        print(f"[Path Conversion] Converted {replacements_made} relative paths to absolute paths")
        print(f"[Path Conversion] Base path: {base_path}")

    return modified_code


def add_path_resolver_to_function(code: str, base_path: str) -> str:
    """
    Add path resolver to function to allow relative paths and automatically convert them to absolute paths

    Args:
        code: Python code string
        base_path: Base path for Docker executor work directory

    Returns:
        str: Code with path resolver added
    """
    if not isinstance(code, str):
        return code

    path_resolver_code = f'''
def _resolve_data_path(path: str, base_path: str = "{base_path}") -> str:
    import os

    if path.startswith('./'):
        return os.path.join(base_path, path[2:])
    elif path.startswith('/'):
        return path
    else:
        return os.path.join(base_path, path)

'''


    import re
    path_usage_patterns = [
        (r'(pd\.read_\w+)\s*\(\s*(["\']\.\/[^"\']*["\'])',
         lambda m: f'{m.group(1)}(_resolve_data_path({m.group(2)})'),
        (r'open\s*\(\s*(["\']\.\/[^"\']*["\'])',
         lambda m: f'open(_resolve_data_path({m.group(1)})'),
        (r'os\.path\.join\s*\(\s*(["\']\.\/[^"\']*["\'])',
         lambda m: f'os.path.join(_resolve_data_path({m.group(1)})'),
        (r'Path\s*\(\s*(["\']\.\/[^"\']*["\'])',
         lambda m: f'Path(_resolve_data_path({m.group(1)})'),
    ]

    for pattern, replacement_func in path_usage_patterns:
        modified_code = re.sub(pattern, replacement_func, modified_code)

    return modified_code


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
            lang = str(block.get('language', '')).lower()
            code_text = block.get('code', '')
            if lang in ['python', 'py'] and isinstance(code_text, str):
                code_text = sanitize_python_code(code_text)
            block['code'] = code_text
            block['timestamp'] = current_time
            blocks_to_save.append(block)
        else:
            lang = str(getattr(block, 'language', '')).lower()
            code_text = getattr(block, 'code', '')
            if lang in ['python', 'py'] and isinstance(code_text, str):
                code_text = sanitize_python_code(code_text)
            blocks_to_save.append({
                'language': block.language,
                'code': code_text,
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

def save_code_as_functions(code_blocks: List[CodeBlock], description: str, output_file: str = "generated_functions.py",
                          convert_paths: bool = True, add_path_resolver: bool = False,
                          add_uuid_to_functions: bool = True,
                          base_path: str = "./coding/dataset") -> None:
    """
    Convert code blocks to Python functions and save to a file

    Args:
        code_blocks: List of code blocks
        description: Description of the task
        output_file: Output Python file name
        convert_paths: Whether to convert relative paths to absolute paths
        add_path_resolver: Whether to add path resolver function for dynamic path resolution
        add_uuid_to_functions: Whether to add UUID suffix to function names to avoid conflicts
        base_path: Base path for Docker executor work directory
    """
    from datetime import datetime
    
    function_template = '''
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

            # Sanitize code to remove unguarded invocations
            sanitized = sanitize_python_code(block.code)

            # Add UUID to function names if requested
            function_mapping = {}
            if add_uuid_to_functions:
                sanitized, function_mapping = add_uuid_to_function_names(sanitized)
                if function_mapping:
                    print(f"[UUID] Added UUID to functions: {function_mapping}")

            # Convert relative paths to absolute paths if requested
            if convert_paths:
                sanitized = convert_relative_to_absolute_paths(sanitized, base_path)
            elif add_path_resolver:
                sanitized = add_path_resolver_to_function(sanitized, base_path)

            # Indent code (no additional indentation needed per current template)
            indented_code = "\n".join("" + line for line in sanitized.split("\n"))

            # Generate function content
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
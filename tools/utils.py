def _resolve_data_path(path: str, base_path: str = "{base_path}") -> str:
    """
    解析数据路径，支持相对路径和绝对路径

    Args:
        path: 输入路径，可以是相对路径（如 './Bank/telemetry'）或绝对路径
        base_path: 数据目录基础路径

    Returns:
        str: 解析后的绝对路径
    """
    import os

    if path.startswith('./'):
        # 相对路径，转换为绝对路径
        return os.path.join(base_path, path[2:])
    elif path.startswith('/'):
        # 已经是绝对路径，直接返回
        return path
    else:
        # 不以 ./ 或 / 开头，假设是相对于数据目录的路径
        return os.path.join(base_path, path)


def _analyze_function_calls(module):
    """
    分析模块中的函数调用关系，找出被其他函数调用的函数

    Args:
        module: 要分析的模块对象

    Returns:
        set: 被其他函数调用的函数名集合
    """
    import ast
    import inspect
    import traceback

    try:
        # 获取模块的源代码
        source = inspect.getsource(module)

        # 解析AST
        tree = ast.parse(source)

        # 获取模块中定义的所有函数名
        defined_functions = set()
        called_functions = set()

        class FunctionAnalyzer(ast.NodeVisitor):
            def __init__(self):
                self.current_function = None

            def visit_FunctionDef(self, node):
                # 记录函数定义
                defined_functions.add(node.name)
                old_function = self.current_function
                self.current_function = node.name
                self.generic_visit(node)
                self.current_function = old_function

            def visit_Call(self, node):
                # 记录函数调用
                if self.current_function:  # 只记录在函数内部的调用
                    if isinstance(node.func, ast.Name):
                        # 直接函数调用，如 func()
                        called_functions.add(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        # 方法调用，如 obj.method()，这里我们不关心
                        pass
                self.generic_visit(node)

        analyzer = FunctionAnalyzer()
        analyzer.visit(tree)

        # 返回被调用且在模块中定义的函数
        return called_functions.intersection(defined_functions)

    except Exception as e:
        print(f"Error analyzing function calls: {e}")
        print(traceback.format_exc())
        return set()


def get_generated_functions_excluding_called(module, exclude_patterns=None):
    """
    获取模块中动态生成的函数，排除被其他函数调用的函数

    Args:
        module: 要分析的模块对象
        exclude_patterns: 要排除的函数名模式列表，默认包含常见的辅助函数

    Returns:
        list: 不被其他函数调用的函数对象列表
    """
    import sys
    import inspect
    import traceback

    try:
        if exclude_patterns is None:
            exclude_patterns = [
                "_resolve_data_path",
                "get_metricCoder_generated_functions",
                "get_logCoder_generated_functions",
                "get_traceCoder_generated_functions",
                "get_generated_functions_excluding_called"  # 排除自身
            ]

        # 分析函数调用关系
        called_functions = _analyze_function_calls(module)

        generated_funcs = []
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj):
                # 排除指定模式的函数
                if name in exclude_patterns:
                    continue
                # 排除被其他函数调用的函数
                if name in called_functions:
                    continue
                # 排除私有函数（以_开头）
                if name.startswith('_'):
                    continue
                # 排除从其他模块导入的函数（检查模块名）
                if hasattr(obj, '__module__') and obj.__module__ != module.__name__:
                    continue

                generated_funcs.append(obj)

        return generated_funcs

    except Exception as e:
        print(f"Error getting generated functions: {e}")
        print(traceback.format_exc())
        return []



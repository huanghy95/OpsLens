import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="flaml")
import asyncio

from dataclasses import dataclass

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from prompt import get_prompt_module
from config_manager import get_prompt_type_with_env_override, get_tool_ecosystem_paths, get_tool_files, get_openai_config, get_model_clients_config

from tool_ecosystem import ToolEcosystemManager, ToolExecutionRecord
import traceback
import importlib, inspect
import logging

prompt_type = get_prompt_type_with_env_override()
prompt_module = get_prompt_module(prompt_type)





def get_dynamic_tool_functions():
    tool_files = get_tool_files()

    try:
        # import metric tools
        metric_module = importlib.import_module(tool_files["metric_tools"])
        get_metricCoder_generated_functions = getattr(metric_module, "get_metricCoder_generated_functions", lambda: [])

        # import log tools
        log_module = importlib.import_module(tool_files["log_tools"])
        get_logCoder_generated_functions = getattr(log_module, "get_logCoder_generated_functions", lambda: [])

        # import trace tools
        trace_module = importlib.import_module(tool_files["trace_tools"])
        get_traceCoder_generated_functions = getattr(trace_module, "get_traceCoder_generated_functions", lambda: [])

        return get_metricCoder_generated_functions, get_logCoder_generated_functions, get_traceCoder_generated_functions

    except ImportError as e:
        logging.debug(f"[Dynamic Tool Import] warning: Failed to import tool modules: {e}")
        logging.debug(f"[Dynamic Tool Import] Falling back to default tool modules")

        try:
            from tools.bank_metric_tools import get_metricCoder_generated_functions
            from tools.bank_log_tools import get_logCoder_generated_functions
            from tools.bank_trace_tools import get_traceCoder_generated_functions
            return get_metricCoder_generated_functions, get_logCoder_generated_functions, get_traceCoder_generated_functions
        except ImportError:
            # if default modules also fail, return empty functions
            return lambda: [], lambda: [], lambda: []

get_metricCoder_generated_functions, get_logCoder_generated_functions, get_traceCoder_generated_functions = get_dynamic_tool_functions()

system_planer = getattr(prompt_module, "system_planer", "")
system_investigator = getattr(prompt_module, "system_investigator", "")
system_explorer = getattr(prompt_module, "system_explorer", "")
system_reasoner = getattr(prompt_module, "system_reasoner", "")


@dataclass
class DiagnosisEvent:
    data_source: str  # Metric, Log, or Trace
    start_time: str
    end_time: str
    component: str
    description: str

@dataclass
class RootCause:
    summary: str
    start_time: str
    end_time: str
    component: str
    description: str


def create_model_client(client_type: str = "default"):
    """Create model client based on configuration"""
    llm_config = get_openai_config()
    model_clients_config = get_model_clients_config()

    api_key = llm_config.get("api_key", "")
    base_url = llm_config.get("base_url", "")

    client_config = model_clients_config.get(client_type, model_clients_config.get("default", {}))

    # Prepare client parameters
    client_params = {
        "model": client_config.get("model", "gpt-5-mini"),
        "api_key": api_key,
        "model_info": client_config.get("model_info", {
            "vision": True,
            "function_calling": True,
            "json_output": False,
            "family": "unknown"
        }),
        "temperature": client_config.get("temperature", 0),
    }

    # Only add base_url if it exists and is not empty
    if base_url:
        client_params["base_url"] = base_url

    return OpenAIChatCompletionClient(**client_params)

# Create model clients
reason_model_client = create_model_client("reason")
model_client = create_model_client("default")




# ---------------- Tool Ecosystem Integration ----------------
# Three managers and execution chains by domain
tool_ecosystem_paths = get_tool_ecosystem_paths()
_metric_tool_manager = ToolEcosystemManager(storage_dir=tool_ecosystem_paths["metric"])
_log_tool_manager = ToolEcosystemManager(storage_dir=tool_ecosystem_paths["log"])
_trace_tool_manager = ToolEcosystemManager(storage_dir=tool_ecosystem_paths["trace"])

try:
    from config_manager import get_tool_files_with_env_override
    tool_files = get_tool_files_with_env_override()

    _metric_tool_manager.sync_tools_from_module(tool_files["metric_tools"], tags=["metric"], data_source="metric", author="bootstrap")
    _log_tool_manager.sync_tools_from_module(tool_files["log_tools"], tags=["log"], data_source="log", author="bootstrap")
    _trace_tool_manager.sync_tools_from_module(tool_files["trace_tools"], tags=["trace"], data_source="trace", author="bootstrap")
    # Fallback bootstrap if none synced (e.g., import issues in manager context)
    def _bootstrap_register_public_functions(manager: ToolEcosystemManager, module_path: str, data_source: str):
        try:
            module = importlib.import_module(module_path)
            for name, fn in inspect.getmembers(module, inspect.isfunction):
                if name.startswith('_'):
                    continue
                excluded_function_patterns = [
                    'get_metricCoder_generated_functions',
                    'get_logCoder_generated_functions',
                    'get_traceCoder_generated_functions',
                    'get_generated_functions',  
                    'get_generated_functions_excluding_called',  
                ]

                if name in excluded_function_patterns:
                    logging.info(f"[Bootstrap] Skipping excluded helper function: {name}")
                    continue

                if name.endswith('Coder_generated_functions') or name.endswith('_generated_functions'):
                    logging.info(f"[Bootstrap] Skipping helper function by pattern: {name}")
                    continue

                exists = any((getattr(t, 'module_path', None) == module_path and getattr(t, 'function_name', None) == name) for t in manager.tools.values())
                if exists:
                    continue
                desc = (inspect.getdoc(fn) or '').splitlines()[0].strip() if inspect.getdoc(fn) else ''
                try:
                    manager.add_new_tool(
                        name=f"{module_path}_{name}",
                        description=desc,
                        function_name=name,
                        tags=[module_path, data_source, "bootstrap"],
                        data_source=data_source,
                        author="bootstrap",
                        module_path=module_path,
                        task_description=""  
                    )
                except Exception as e:
                    logging.debug(f"Error bootstrapping tool {module_path}.{name}: {e}")
        except Exception as e:
            logging.debug(f"Error loading module {module_path} for bootstrap: {e}")
            logging.debug(traceback.format_exc())
    if len(_metric_tool_manager.tools) == 0:
        _bootstrap_register_public_functions(_metric_tool_manager, tool_files["metric_tools"], "metric")
    if len(_log_tool_manager.tools) == 0:
        _bootstrap_register_public_functions(_log_tool_manager, tool_files["log_tools"], "log")
    if len(_trace_tool_manager.tools) == 0:
        _bootstrap_register_public_functions(_trace_tool_manager, tool_files["trace_tools"], "trace")
except Exception as e:
    logging.debug(f"Error syncing tools from modules: {e}")
    logging.debug(traceback.format_exc())

_metric_tool_chain = []
_log_tool_chain = []
_trace_tool_chain = []

_unified_tool_chain = []

# format:{(module_path, function_name): is_success}
_tool_call_results = {}

def get_metric_tool_ecosystem_manager() -> ToolEcosystemManager:
    return _metric_tool_manager

def get_log_tool_ecosystem_manager() -> ToolEcosystemManager:
    return _log_tool_manager

def get_trace_tool_ecosystem_manager() -> ToolEcosystemManager:
    return _trace_tool_manager

def get_metric_tool_chain() -> list:
    return list(_metric_tool_chain)

def get_log_tool_chain() -> list:
    return list(_log_tool_chain)

def get_trace_tool_chain() -> list:
    return list(_trace_tool_chain)

def get_unified_tool_chain() -> list:
    return list(_unified_tool_chain)

def get_tool_call_results() -> dict:
    return dict(_tool_call_results)

def reset_tool_chains():
    _metric_tool_chain.clear()
    _log_tool_chain.clear()
    _trace_tool_chain.clear()
    _unified_tool_chain.clear()
    _tool_call_results.clear()

def record_tool_call(explorer_name: str, function_name: str):
    """Append a tool binding to the corresponding domain chain based on explorer name and ensure it is registered in the manager."""
    record_tool_call_with_result(explorer_name, function_name, True)  

def record_tool_call_with_result(explorer_name: str, function_name: str, is_success: bool):
    """Append a tool binding to the corresponding domain chain and record the execution result."""
    try:
        from config_manager import get_tool_files_with_env_override
        tool_files = get_tool_files_with_env_override()

        if explorer_name == "metric_explorer":
            logging.info(f"record_tool_call: {explorer_name}.{function_name}, success: {is_success}")
            module_path = tool_files["metric_tools"]
            manager = _metric_tool_manager
            data_source = "metric"
            _metric_tool_chain.append((module_path, function_name))
        elif explorer_name == "log_explorer":
            logging.info(f"record_tool_call: {explorer_name}.{function_name}, success: {is_success}")
            module_path = tool_files["log_tools"]
            manager = _log_tool_manager
            data_source = "log"
            _log_tool_chain.append((module_path, function_name))
        elif explorer_name == "trace_explorer":
            logging.info(f"record_tool_call: {explorer_name}.{function_name}, success: {is_success}")
            module_path = tool_files["trace_tools"]
            manager = _trace_tool_manager
            data_source = "trace"
            _trace_tool_chain.append((module_path, function_name))
        else:
            return

        _tool_call_results[(module_path, function_name)] = is_success

        if is_success:
            tool_id = None
            for tid, tool in manager.tools.items():
                if getattr(tool, 'module_path', None) == module_path and getattr(tool, 'function_name', None) == function_name:
                    tool_id = tid
                    break

            if tool_id:
                _unified_tool_chain.append(tool_id)
                logging.info(f"[Unified Tool Chain] Added successful tool call: {tool_id} ({explorer_name}.{function_name})")
            else:
                logging.info(f"[Unified Tool Chain] Warning: Tool ID not found for {module_path}.{function_name}")
        else:
            logging.info(f"[Unified Tool Chain] Skipped failed tool call: {explorer_name}.{function_name}")

        try:
            from tool_ecosystem import ToolExecutionRecord
            from datetime import datetime

            execution_record = ToolExecutionRecord(
                task_description=f"Tool call from {explorer_name}",
                exit_code=0 if is_success else 1,
                stdout="",
                stderr="" if is_success else "Tool execution failed",
                latency_ms=0.0,  
                is_effective=None,  
                rca_task_score=None,  
                timestamp=datetime.utcnow(),
                is_chain_contribution=False
            )

            manager.update_tool_after_execution_by_binding(module_path, function_name, execution_record)
        except Exception as e:
            logging.debug(f"Error updating tool execution record: {e}")
            logging.debug(traceback.format_exc())

        
        exists = False
        for t in manager.tools.values():
            if getattr(t, 'module_path', None) == module_path and getattr(t, 'function_name', None) == function_name:
                exists = True
                break

        if not exists:
            
            try:
                import importlib, inspect
                module = importlib.import_module(module_path)
                fn = getattr(module, function_name, None)

                if fn is not None:  
                    desc = (inspect.getdoc(fn) or '').splitlines()[0].strip() if inspect.getdoc(fn) else ''
                    try:
                        manager.add_new_tool(
                            name=f"{module_path}_{function_name}",
                            description=desc,
                            function_name=function_name,
                            tags=[module_path, data_source, "runtime"],
                            data_source=data_source,
                            author="runtime",
                            module_path=module_path,
                            task_description=""  
                        )
                        logging.info(f"Registered existing tool: {module_path}.{function_name}")
                    except Exception as e:
                        logging.debug(f"Error registering existing tool: {module_path}.{function_name}: {e}")
                else:
                    logging.info(f"Function {function_name} not found in {module_path}, skipping registration (likely newly generated)")
            except ImportError:
                logging.debug(f"Module {module_path} not found, skipping tool registration")
            except Exception as e:
                logging.debug(f"Error checking module {module_path} for function {function_name}: {e}")
    except Exception as e:
        logging.debug(f"Error recording tool call for {explorer_name}.{function_name}: {e}")
        logging.debug(traceback.format_exc())

# Planner Agent
planner_agent = AssistantAgent(
    name="planner",
    system_message=system_planer,
    model_client=model_client,
)

# Investigator Agent (Controller)
investigator_agent = AssistantAgent(
    name="investigator",
    system_message=system_investigator,
    model_client=model_client,
)


metric_explorer = AssistantAgent(
    name="metric_explorer",
    description="Agent for exploring metric data",
    system_message="You are the metric explorer "+system_explorer,
    model_client=model_client,
    tools=get_metricCoder_generated_functions(),
)


log_explorer = AssistantAgent(
    name="log_explorer",
    description="Agent for exploring log data",
    system_message="You are the log explorer "+system_explorer,
    model_client=model_client,
    tools=get_logCoder_generated_functions(),
)


trace_explorer = AssistantAgent(
    name="trace_explorer",
    description="Agent for exploring trace data",
    system_message="You are the trace explorer "+system_explorer,
    model_client=model_client,
    tools=get_traceCoder_generated_functions(),
)


# Reasoner Agent
reasoner_agent = AssistantAgent(
    name="reasoner",
    system_message=system_reasoner,
    model_client=reason_model_client,
)





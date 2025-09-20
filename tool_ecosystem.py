import uuid
import math
import pickle
import json
import chromadb
import networkx as nx
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict
import traceback
import importlib
import inspect
import io
from contextlib import redirect_stdout
import openai
from typing import List
from config_manager import get_openai_config
import logging

def wilson_score_lower_bound(pos, n, confidence=0.95):
    """
    Calculate the lower bound of Wilson confidence interval - scoring algorithm to avoid small sample bias
    :param pos: positive outcomes
    :param n: total trials
    :param confidence: confidence level (e.g., 0.95 for 95% confidence)
    :return: a score between 0 and 1
    """
    if n == 0:
        return 0.0

    # Parameter validation: ensure pos <= n to avoid mathematical domain errors
    if pos > n:
        logging.info(f"Warning: pos ({pos}) > n ({n}), adjusting pos to n")
        logging.debug(traceback.format_exc())
        pos = n

    if pos < 0:
        logging.info(f"Warning: pos ({pos}) < 0, adjusting pos to 0")
        pos = 0

    if n < 0:
        logging.info(f"Warning: n ({n}) < 0, returning 0.0")
        return 0.0

    # Z-score for the given confidence level
    z = {0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)

    p_hat = pos / n

    try:
        # Ensure the value under square root is non-negative
        sqrt_term = (p_hat*(1-p_hat) + z*z/(4*n))/n
        if sqrt_term < 0:
            logging.info(f"Warning: sqrt_term ({sqrt_term}) < 0, using 0 instead")
            sqrt_term = 0

        numerator = p_hat + (z*z / (2*n)) - z * math.sqrt(sqrt_term)
        denominator = 1 + (z*z / n)

        result = max(0.0, numerator / denominator)
        return min(1.0, result)  # Ensure result does not exceed 1.0

    except Exception as e:
        logging.debug(f"Error in wilson_score_lower_bound: pos={pos}, n={n}, p_hat={p_hat}")
        logging.debug(f"Exception: {e}")
        logging.debug(traceback.format_exc())
        return 0.0

@dataclass
class ToolExecutionRecord:
    """Tool execution record - records detailed information for each tool execution"""
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_description: str = ""
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    latency_ms: float = 0.0
    is_effective: Optional[bool] = None  # Considered effective if current RCA task succeeds, compared with GT at the end
    rca_task_score: Optional[float] = None  # Final score of this RCA task
    feedback_notes: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    is_chain_contribution: bool = False  # Whether it's a task chain contribution record (not counted in runs)

@dataclass
class ToolQualityMetrics:
    """Tool quality metrics - implements core leaderboard functionality"""
    total_runs: int = 0
    success_runs: int = 0  # exit_code == 0
    effective_runs: int = 0  # Considered effective if current RCA task succeeds
    accumulated_task_score: float = 0.0  # Accumulated final RCA scores of all tasks that called this tool
    total_latency_ms: float = 0.0
    last_used_timestamp: Optional[datetime] = None
    last_success_timestamp: Optional[datetime] = None
    created_timestamp: datetime = field(default_factory=datetime.utcnow)
    # SBFL (Spectrum-Based Fault Localization) related fields
    onp_count: int = 0  # Number of test cases that did not execute this element and passed (Not Executed & Passed)

    @property
    def success_rate(self) -> float:
        """Success rate - basic stability indicator"""
        return (self.success_runs / self.total_runs) if self.total_runs > 0 else 0.0

    @property
    def effectiveness_rate(self) -> float:
        """Effectiveness rate - what proportion of successful runs are effective, using Wilson confidence interval to avoid small sample bias"""
        # Ensure data consistency: effective_runs should not exceed success_runs
        effective_runs = min(self.effective_runs, self.success_runs)

        if effective_runs != self.effective_runs:
            logging.info(f"Warning: effective_runs ({self.effective_runs}) > success_runs ({self.success_runs}), adjusting to {effective_runs}")

        return wilson_score_lower_bound(effective_runs, self.success_runs)

    @property
    def avg_task_score(self) -> float:
        """Average task contribution - tool's contribution to end-to-end tasks"""
        return (self.accumulated_task_score / self.total_runs) if self.total_runs > 0 else 0.0

    @property
    def average_latency_ms(self) -> float:
        """Average latency"""
        return (self.total_latency_ms / self.success_runs) if self.success_runs > 0 else 0.0

    # SBFL (Spectrum-Based Fault Localization) related properties
    @property
    def oef(self) -> int:
        """Oef (Executed & Failed): Number of test cases that executed this element and failed"""
        return self.total_runs - self.success_runs

    @property
    def oep(self) -> int:
        """Oep (Executed & Passed): Number of test cases that executed this element and passed"""
        return self.effective_runs

    @property
    def onp(self) -> int:
        """Onp (Not Executed & Passed): Number of test cases that did not execute this element and passed"""
        return self.onp_count

    @property
    def dstar2_score(self) -> float:
        """Dstar2 score: Oep^2 / (Oep + Onp)"""
        if self.oep + self.onp == 0:
            return 0.0
        return (self.oep ** 2) / (self.oep + self.onp)

    def validate_and_fix_metrics(self):
        """Validate and fix metric data consistency"""
        # Ensure effective_runs <= success_runs
        if self.effective_runs > self.success_runs:
            logging.debug(f"Warning: Fixing inconsistent metrics - effective_runs ({self.effective_runs}) > success_runs ({self.success_runs})")
            self.effective_runs = self.success_runs

        # Ensure success_runs <= total_runs
        if self.success_runs > self.total_runs:
            logging.debug(f"Warning: Fixing inconsistent metrics - success_runs ({self.success_runs}) > total_runs ({self.total_runs})")
            self.success_runs = self.total_runs
            self.effective_runs = min(self.effective_runs, self.success_runs)

        # Ensure all counts are non-negative
        self.total_runs = max(0, self.total_runs)
        self.success_runs = max(0, self.success_runs)
        self.effective_runs = max(0, self.effective_runs)

    @property
    def score(self) -> float:
        """Leaderboard core algorithm - Dstar2 score based on spectrum analysis SBFL"""
        # Validate data consistency before calculating score
        self.validate_and_fix_metrics()

        if self.total_runs == 0:
            return 0.0

        # Use Dstar2 formula: Oep^2 / (Oep + Onp)
        dstar2_base_score = self.dstar2_score

        return dstar2_base_score

    @property
    def unified_score(self)->float:
        denominator = self.oep + self.oef
        if denominator == 0:
            # Handle case where denominator is zero, can return a default value or throw a more explicit exception
            return 0  # or other appropriate default value
            # or raise ValueError("oep and oef cannot both be zero")
        return self.oep / denominator

@dataclass
class Tool:
    """Core data structure of tool - digital life form of the tool"""
    # --- Static identity attributes ---
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    function_name: str = ""  # Function name 
    function_signature: Optional[str] = None  # Function signature (can be dynamically parsed and cached as needed)
    function_signature_last_resolved_at: Optional[datetime] = None
    module_path: str = "generated_functions"  # Bound module path (e.g., 'generated_functions')
    author: str = "system"  # e.g., "LLM-v2.1", "user-refined-v2"
    version: int = 1
    created_at: datetime = field(default_factory=datetime.utcnow)

    # --- Semantic and classification attributes ---
    description: str = ""  # Detailed description of the tool itself
    task_description: str = ""  # Task description when executing
    intent_embedding: List[float] = field(default_factory=list)  # Vector generated from task_description+description
    tags: List[str] = field(default_factory=list)
    data_source: str = ""  # "metric", "log", "trace"

    # --- Dynamic quality and lifecycle attributes ---
    metrics: ToolQualityMetrics = field(default_factory=ToolQualityMetrics)
    execution_history: List[ToolExecutionRecord] = field(default_factory=list, repr=False)
    status: str = "active"  # "active", "deprecated", "archived"

class ToolGraph:
    """
    Tool graph - maintains contextual relationships between tools
    - Nodes: generated tools
    - Node weights: quality scores
    - Edges: tool chain sequence, directed edges, A→B indicates that B's call immediately follows A's call in a query
    - Edge weights: occur_rate, probability that tool context and its binding appear and current task succeeds
    """

    def __init__(self):
        # Build directed graph using NetworkX
        self.graph = nx.DiGraph()
        # Record the number of occurrences of each <A,B> combination
        self.edge_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        # Record the number of successful occurrences of each <A,B> combination
        self.edge_success_counts: Dict[Tuple[str, str], int] = defaultdict(int)

    def add_or_update_edge(self, source_tool_id: str, target_tool_id: str, task_succeeded: bool):
        """In an RCA task's tool_chain, when B is called immediately after A, update the edge between them"""
        edge_key = (source_tool_id, target_tool_id)

        # Update statistics
        self.edge_counts[edge_key] += 1
        if task_succeeded:
            self.edge_success_counts[edge_key] += 1

        # Calculate success rate as edge weight
        success_rate = self.edge_success_counts[edge_key] / self.edge_counts[edge_key]

        # Update edge in graph
        if self.graph.has_edge(source_tool_id, target_tool_id):
            self.graph[source_tool_id][target_tool_id]['weight'] = success_rate
            self.graph[source_tool_id][target_tool_id]['count'] = self.edge_counts[edge_key]
            self.graph[source_tool_id][target_tool_id]['success_count'] = self.edge_success_counts[edge_key]
        else: # First occurrence, initialize edge
            self.graph.add_edge(source_tool_id, target_tool_id,
                              weight=success_rate,
                              count=self.edge_counts[edge_key],
                              success_count=self.edge_success_counts[edge_key])

    def get_successors(self, tool_id: str, min_weight: float = 0.5) -> Dict[str, float]:
        """Get successors of a tool and their weights (filter low-weight edges)"""
        if not self.graph.has_node(tool_id):
            return {}

        successors = {}
        for successor in self.graph.successors(tool_id):
            weight = self.graph[tool_id][successor]['weight']
            if weight >= min_weight:
                successors[successor] = weight

        # Sort by weight in descending order
        return dict(sorted(successors.items(), key=lambda x: x[1], reverse=True))

    def get_parent_tools(self, tool_id: str, min_weight: float = 0.5) -> List[str]:
        """Get parent tools of a tool"""
        return [node for node in self.graph.predecessors(tool_id) if self.graph.has_edge(node, tool_id) and self.graph[node][tool_id]['weight'] >= min_weight]

    def get_context_hint(self, source_tool_id: str, target_tool_id: str, source_tool_name: Optional[str] = None) -> Optional[str]:
        """Generate context hint for specific tool pair, prioritize displaying tool name."""
        if not self.graph.has_edge(source_tool_id, target_tool_id):
            return None

        edge_data = self.graph[source_tool_id][target_tool_id]
        weight = edge_data['weight']
        count = edge_data['count']

        name = source_tool_name or source_tool_id
        if count >= 3 and weight >= 0.6:  # At least 3 calls and success rate >= 60%
            return f"Context Hint: After using '{name}', this tool is often used next with {weight:.0%} success rate (based on {count} historical cases)."
        # Unified text for other cases to avoid redundant branches
        return f"Context Hint: This tool has been used after '{name}' before with {weight:.0%} success rate."

    
    

    def save_to_file(self, file_path: str):
        """Save graph to file"""
        try:
            data = {
                'edge_counts': dict(self.edge_counts),
                'edge_success_counts': dict(self.edge_success_counts),
                'graph_data': nx.node_link_data(self.graph, edges="edges")
            }
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logging.debug(f"Error saving tool graph: {e}")
            logging.debug(traceback.format_exc())

    def load_from_file(self, file_path: str):
        """Load graph from file"""
        try:
            if not Path(file_path).exists():
                return

            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            self.edge_counts = defaultdict(int, {eval(k) if isinstance(k, str) else k: v
                                               for k, v in data['edge_counts'].items()})
            self.edge_success_counts = defaultdict(int, {eval(k) if isinstance(k, str) else k: v
                                                       for k, v in data['edge_success_counts'].items()})
            graph_payload = data['graph_data']
            # Historical compatibility: try edges key first, otherwise use default links key
            try:
                if isinstance(graph_payload, dict) and 'edges' in graph_payload:
                    self.graph = nx.node_link_graph(graph_payload, edges="edges")
                else:
                    self.graph = nx.node_link_graph(graph_payload)
            except Exception:
                # Fallback: try again with default parameters
                self.graph = nx.node_link_graph(graph_payload)
        except Exception as e:
            logging.debug(f"Error loading tool graph: {e}")
            logging.debug(traceback.format_exc())

    def remove_tool(self, tool_id: str):
        """Remove a tool node and related edges from the graph, and clean up statistics."""
        try:
            if self.graph.has_node(tool_id):
                self.graph.remove_node(tool_id)
            # Clean up edge statistics
            try:
                self.edge_counts = defaultdict(int, {k: v for k, v in self.edge_counts.items() if k[0] != tool_id and k[1] != tool_id})
                self.edge_success_counts = defaultdict(int, {k: v for k, v in self.edge_success_counts.items() if k[0] != tool_id and k[1] != tool_id})
            except Exception as e:
                logging.debug(f"Error pruning edge stats for removed tool {tool_id}: {e}")
                logging.debug(traceback.format_exc())
        except Exception as e:
            logging.debug(f"Error removing tool node {tool_id} from graph: {e}")
            logging.debug(traceback.format_exc())

class ToolEcosystemManager:
    """Tool ecosystem manager - core brain of the system"""

    # Class-level shared tool graph, all instances share the same graph
    _shared_tool_graph = None
    _shared_graph_file = None

    def __init__(self, storage_dir: str = "./tool_ecosystem"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

        # Initialize vector database
        self.chroma_client = chromadb.PersistentClient(path=str(self.storage_dir / "chroma_db"))
        self.tool_collection = self.chroma_client.get_or_create_collection(
            name="tool_embeddings",
            metadata={"hnsw:space": "cosine"}
        )

        # Initialize shared tool graph (if not already initialized)
        if ToolEcosystemManager._shared_tool_graph is None:
            ToolEcosystemManager._shared_tool_graph = ToolGraph()
            # Use the first instance's storage directory as the shared graph storage location
            ToolEcosystemManager._shared_graph_file = self.storage_dir.parent / "shared_tool_graph.pkl"
            ToolEcosystemManager._shared_tool_graph.load_from_file(str(ToolEcosystemManager._shared_graph_file))

        # Reference shared tool graph
        self.tool_graph = ToolEcosystemManager._shared_tool_graph
        self.graph_file = ToolEcosystemManager._shared_graph_file

        # Tool storage (ID -> Tool object mapping)
        self.tools: Dict[str, Tool] = {}
        self.tools_file = self.storage_dir / "tools.pkl"
        self._load_tools()

        # Configuration parameters
        self.SEMANTIC_THRESHOLD = 0.75  # Semantic relevance threshold

    # ---------- Binding and signature parsing utility functions ----------
    def resolve_callable(self, function_name: str, module_path: str = "generated_functions"):
        """Resolve to callable object based on module and function name."""
        try:
            module = importlib.import_module(module_path)
            # Prevent hot update desynchronization
            module = importlib.reload(module)
            func = getattr(module, function_name, None)
            if not callable(func):
                return None
            return func
        except Exception as e:
            logging.debug(f"Error resolving callable {module_path}.{function_name}: {e}")
            logging.debug(traceback.format_exc())
            return None

    def get_function_signature(self, function_name: str, module_path: str = "generated_functions") -> str:
        """Get function signature string, e.g., function_x(a: int, b: str) -> None. Returns empty string on failure."""
        try:
            func = self.resolve_callable(function_name, module_path)
            if func is None:
                return ""
            sig = str(inspect.signature(func))
            return f"{function_name}{sig}"
        except Exception as e:
            logging.debug(f"Error getting function signature for {module_path}.{function_name}: {e}")
            logging.debug(traceback.format_exc())
            return ""

    def extract_function_info_from_module(self, module_path: str) -> List[Dict[str, Any]]:
        """Extract all compliant function names, signatures, and description information from specified module."""
        infos: List[Dict[str, Any]] = []
        try:
            module = None
            try:
                module = importlib.import_module(module_path)
                module = importlib.reload(module)
            except Exception:
                module = None
            # Fallback: load by file path if regular import fails
            if module is None:
                try:
                    from importlib import util as importlib_util
                    fs_path = (Path.cwd() / (module_path.replace('.', '/') + '.py')).resolve()
                    if fs_path.exists():
                        spec = importlib_util.spec_from_file_location(module_path, str(fs_path))
                        if spec and spec.loader:
                            module = importlib_util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                except Exception:
                    module = None
            if module is None:
                logging.debug(f"Error: cannot import module {module_path}")
                return []

            # Use intelligent function call analysis to get functions that should be exposed
            functions_to_include = []

            # Check if it's a tool module, if so use corresponding get_*Coder_generated_functions
            # Dynamically import correct tool module based on configuration
            tool_type = None
            if module_path in ['tools.metric_tools', 'metric_tools'] or 'metric_tools' in module_path:
                tool_type = 'metric_tools'
                func_name = 'get_metricCoder_generated_functions'
            elif module_path in ['tools.log_tools', 'log_tools'] or 'log_tools' in module_path:
                tool_type = 'log_tools'
                func_name = 'get_logCoder_generated_functions'
            elif module_path in ['tools.trace_tools', 'trace_tools'] or 'trace_tools' in module_path:
                tool_type = 'trace_tools'
                func_name = 'get_traceCoder_generated_functions'

            if tool_type:
                try:
                    # Get correct tool module path based on configuration
                    from config_manager import get_config_manager
                    cm = get_config_manager()
                    tool_files = cm.get_tool_files()

                    # Get module path for corresponding tool type
                    target_module_path = tool_files.get(tool_type, module_path)

                    # Dynamically import module
                    target_module = importlib.import_module(target_module_path)
                    target_module = importlib.reload(target_module)

                    # Get function and call it
                    if hasattr(target_module, func_name):
                        get_functions = getattr(target_module, func_name)
                        functions_to_include = get_functions()
                        logging.info(f"[Tool Ecosystem] Using smart analysis for {tool_type}: {len(functions_to_include)} functions")
                    else:
                        logging.info(f"[Tool Ecosystem] Warning: {target_module_path} does not have {func_name}")
                        functions_to_include = []

                except Exception as e:
                    logging.debug(f"[Tool Ecosystem] Error using smart analysis for {tool_type}: {e}")
                    logging.debug(traceback.format_exc())
                    functions_to_include = []
            else:
                # For other modules, use generic intelligent analysis
                try:
                    from tools.utils import get_generated_functions_excluding_called
                    functions_to_include = get_generated_functions_excluding_called(module)
                    logging.info(f"[Tool Ecosystem] Using generic smart analysis for {module_path}: {len(functions_to_include)} functions")
                except Exception as e:
                    logging.debug(f"[Tool Ecosystem] Error using generic smart analysis for {module_path}: {e}")
                    functions_to_include = []

            # If intelligent analysis fails, fall back to original logic
            if not functions_to_include:
                logging.info(f"[Tool Ecosystem] Falling back to legacy logic for {module_path}")
                for name, obj in inspect.getmembers(module, inspect.isfunction):
                    if name.startswith('_'):
                        continue

                    # Exclude helper functions in various tool modules
                    # These functions are used to get information about other functions in the module and should not be synced as tools
                    excluded_function_patterns = [
                        'get_metricCoder_generated_functions',
                        'get_logCoder_generated_functions',
                        'get_traceCoder_generated_functions',
                        'get_generated_functions',  # Generic pattern
                        'get_generated_functions_excluding_called',  # Newly added generic function
                    ]

                    if name in excluded_function_patterns:
                        logging.info(f"[Tool Ecosystem] Skipping excluded helper function: {name}")
                        continue

                    # Can also exclude through pattern matching
                    if name.endswith('Coder_generated_functions') or name.endswith('_generated_functions'):
                        logging.info(f"[Tool Ecosystem] Skipping helper function by pattern: {name}")
                        continue

                    functions_to_include.append(obj)

            # Process the obtained function list
            for obj in functions_to_include:
                name = obj.__name__

                try:
                    sig = str(inspect.signature(obj))
                except Exception:
                    sig = "()"
                # Parse description
                desc = ""
                doc = inspect.getdoc(obj) or ""
                if doc:
                    lines = doc.splitlines()
                    # Prioritize parsing lines containing 'Tool description:'
                    for i, line in enumerate(lines):
                        line_stripped = line.strip()
                        if line_stripped.lower().startswith('tool description:'):
                            # If there's content after Tool description:, use it
                            desc_part = line_stripped.split(':', 1)[1].strip()
                            if desc_part:
                                desc = desc_part
                            else:
                                # If there's no content after Tool description:, check the next line
                                if i + 1 < len(lines):
                                    next_line = lines[i + 1].strip()
                                    if next_line:
                                        desc = next_line
                            break
                    if not desc:
                        # Fall back to first line of docstring
                        desc = doc.splitlines()[0].strip()
                infos.append({
                    "function_name": name,
                    "function_signature": f"{name}{sig}",
                    "description": desc,
                    "module_path": module_path
                })
        except Exception as e:
            logging.debug(f"Error extracting functions from module {module_path}: {e}")
            logging.debug(traceback.format_exc())
        return infos

    def extract_function_info_from_generated_functions(self) -> List[Dict[str, Any]]:
        """Extract all function names, signatures, and description information from generated_functions.py."""
        return self.extract_function_info_from_module("generated_functions")

    def sync_tools_from_module(self, module_path: str, tags: Optional[List[str]] = None, data_source: str = "unknown", author: str = "bootstrap", prune_stale: bool = True):
        """Sync tools from specified module, existing function names will be skipped. Optional: clean up historical tools that don't exist in the module."""
        try:
            infos = self.extract_function_info_from_module(module_path)

            existing_items = {tid: t for tid, t in self.tools.items() if getattr(t, 'module_path', None) == module_path}
            existing_func_names = {t.function_name for t in existing_items.values()}
            live_func_names = {info["function_name"] for info in infos}

            if prune_stale:
                removed = 0
                for tid, t in list(existing_items.items()):
                    if t.function_name not in live_func_names:
                        if self.remove_tool(tid):
                            removed += 1
                logging.info(f"[Tool Ecosystem] Pruned {removed} stale tools from {module_path}")
            if not infos:
                logging.info(f"[Tool Ecosystem] No functions found to sync from {module_path}")
                return
            added = 0
            for info in infos:
                if info["function_name"] in existing_func_names or "generated_functions" in info["function_name"]:
                    continue
                tool_id = self.add_new_tool(
                    name=f"{module_path}_{info['function_name']}",
                    description=info.get("description", ""),
                    function_name=info["function_name"],
                    function_signature=info.get("function_signature"),
                    tags=(tags or []) + [module_path, "generated"],
                    data_source=data_source,
                    author=author,
                    module_path=module_path
                )
                if tool_id:
                    added += 1
            logging.info(f"[Tool Ecosystem] Synced {added} tools from {module_path}")

        except Exception as e:
            logging.debug(f"Error syncing tools from module {module_path}: {e}")
            logging.debug(traceback.format_exc())

    def sync_tools_from_module_with_task_context(self, module_path: str, task_description: str,
                                               tags: Optional[List[str]] = None, data_source: str = "unknown",
                                               author: str = "coder", prune_stale: bool = True):
        """Sync tools from specified module and add task description context for new tools"""
        try:
            infos = self.extract_function_info_from_module(module_path)
            if not infos:
                logging.info(f"[Tool Ecosystem] No functions found to sync from {module_path}")
                return

            existing_items = {tid: t for tid, t in self.tools.items() if getattr(t, 'module_path', None) == module_path}
            existing_func_names = {t.function_name for t in existing_items.values()}
            live_func_names = {info["function_name"] for info in infos}
            added = 0

            for info in infos:
                if info["function_name"] in existing_func_names or "generated_functions" in info["function_name"]:
                    continue

                tool_id = self.add_tool_with_task_context(
                    name=f"{module_path}_{info['function_name']}",
                    description=info.get("description", ""),
                    function_name=info["function_name"],
                    task_description=task_description,
                    function_signature=info.get("function_signature"),
                    tags=(tags or []) + [module_path, "generated"],
                    data_source=data_source,
                    author=author,
                    module_path=module_path
                )
                if tool_id:
                    added += 1
                    logging.info(f"Added new tool with task context: {module_path}_{info['function_name']} (ID: {tool_id})")

            logging.info(f"[Tool Ecosystem] Synced {added} tools with task context from {module_path}")

            if prune_stale:
                pruned = 0
                for tool_id, tool in list(existing_items.items()):
                    if tool.function_name not in live_func_names:
                        self.remove_tool(tool_id)
                        pruned += 1
                        logging.info(f"Pruned stale tool: {tool.name} (ID: {tool_id})")
                logging.info(f"[Tool Ecosystem] Pruned {pruned} stale tools from {module_path}")

            self._save_tools()

        except Exception as e:
            logging.debug(f"Error syncing tools with task context from {module_path}: {e}")
            logging.debug(traceback.format_exc())

    def sync_tools_from_generated_functions(self):
        """Cold start/sync: register functions in generated_functions.py as tools (skip existing ones)."""
        try:
            self.sync_tools_from_module("generated_functions")
        except Exception as e:
            logging.debug(f"Error syncing tools from generated_functions: {e}")
            logging.debug(traceback.format_exc())

    def execute_tool_function(self, function_name: str, module_path: str = "generated_functions") -> Dict[str, Any]:
        """Execute function in specified module, capture stdout and exceptions."""
        try:
            func = self.resolve_callable(function_name, module_path)
            if func is None:
                return {"success": False, "stdout": "", "stderr": f"Function {module_path}.{function_name} not found", "error": f"Function {module_path}.{function_name} not found"}
            stdout_buf = io.StringIO()
            with redirect_stdout(stdout_buf):
                result = func()
            return {"success": True, "stdout": stdout_buf.getvalue(), "result": result}
        except Exception as e:
            logging.debug(f"Error executing function {module_path}.{function_name}: {e}")
            logging.debug(traceback.format_exc())
            return {"success": False, "stdout": "", "stderr": str(e), "error": str(e)}

    # ---------- Storage and loading ----------
    def _load_tools(self):
        """Load tools from file"""
        try:
            if self.tools_file.exists():
                with open(self.tools_file, 'rb') as f:
                    self.tools = pickle.load(f)
                # Migration: fill new fields for old version objects
                try:
                    for tool in self.tools.values():
                        if not hasattr(tool, 'module_path'):
                            setattr(tool, 'module_path', 'generated_functions')
                        if not hasattr(tool, 'function_signature'):
                            setattr(tool, 'function_signature', None)
                        if not hasattr(tool, 'function_signature_last_resolved_at'):
                            setattr(tool, 'function_signature_last_resolved_at', None)
                except Exception as e:
                    logging.debug(f"Error migrating loaded tools: {e}")
                    logging.debug(traceback.format_exc())
                logging.info(f"Loaded {len(self.tools)} tools from storage")
                for tool in self.tools.values():
                    logging.info(f"  - {tool.name} (ID: {tool.id})")
        except Exception as e:
            logging.debug(f"Error loading tools: {e}")
            logging.debug(traceback.format_exc())

    def _save_tools(self):
        """Save tools to file"""
        try:
            with open(self.tools_file, 'wb') as f:
                pickle.dump(self.tools, f)
            self.tool_graph.save_to_file(str(self.graph_file))
        except Exception as e:
            logging.debug(f"Error saving tools: {e}")
            logging.debug(traceback.format_exc())

    def add_new_tool(self, name: str, description: str, function_name: str, function_signature: Optional[str] = None,
                     tags: List[str] = None, data_source: str = "", author: str = "system", module_path: str = "generated_functions",
                     task_description: str = "") -> str:
        """Add new tool to ecosystem"""
        try:
            # Create tool object
            resolved_signature = function_signature or self.get_function_signature(function_name, module_path)
            tool = Tool(
                name=name,
                description=description,
                task_description=task_description,
                function_name=function_name,
                function_signature=resolved_signature if resolved_signature else None,
                function_signature_last_resolved_at=datetime.utcnow() if resolved_signature else None,
                module_path=module_path,
                tags=tags or [],
                data_source=data_source,
                author=author
            )

            # Generate embedding (based on combination of task_description + description)
            combined_text = f"{task_description} {description}".strip()
            tool.intent_embedding = self._generate_embedding(combined_text)

            # Store to vector database
            try:
                self.tool_collection.add(
                    embeddings=[tool.intent_embedding],
                    documents=[combined_text],  # Store combination of task_description + description
                    metadatas=[{
                        "tool_id": tool.id,
                        "name": name,
                        "data_source": data_source,
                        "tags": json.dumps(tags or []),
                        "created_at": tool.created_at.isoformat(),
                        "function_name": function_name,
                        "module_path": module_path,
                        "task_description": task_description,
                        "tool_description": description
                    }],
                    ids=[tool.id]
                )
            except Exception as e:
                logging.debug(f"Error adding to vector store for tool {name}: {e}")
                logging.debug(traceback.format_exc())

            # Store to memory
            self.tools[tool.id] = tool

            # Persist
            self._save_tools()

            logging.info(f"Added new tool: {name} (ID: {tool.id})")
            return tool.id

        except Exception as e:
            logging.debug(f"Error adding new tool: {e}")
            logging.debug(traceback.format_exc())
            return ""

    def add_tool_with_task_context(self, name: str, description: str, function_name: str, task_description: str,
                                   function_signature: Optional[str] = None, tags: List[str] = None,
                                   data_source: str = "", author: str = "system", module_path: str = "generated_functions") -> str:
        """Add new tool to ecosystem based on task context"""
        return self.add_new_tool(
            name=name,
            description=description,
            function_name=function_name,
            function_signature=function_signature,
            tags=tags,
            data_source=data_source,
            author=author,
            module_path=module_path,
            task_description=task_description
        )



    def remove_tool(self, tool_id: str, delete_from_vector_store: bool = True) -> bool:
        """Remove a tool, including memory, vector store, graph, then persist."""
        try:
            tool = self.tools.get(tool_id)
            if not tool:
                logging.info(f"Tool {tool_id} not found for removal")
                return False
            # 1) Vector store deletion
            if delete_from_vector_store:
                try:
                    self.tool_collection.delete(ids=[tool_id])
                except Exception as e:
                    logging.debug(f"Error deleting vector entry for tool {tool_id}: {e}")
                    logging.debug(traceback.format_exc())
            # 2) Graph deletion
            try:
                self.tool_graph.remove_tool(tool_id)
            except Exception as e:
                logging.debug(f"Error removing tool {tool_id} from graph: {e}")
                logging.debug(traceback.format_exc())
            # 3) Memory deletion
            try:
                del self.tools[tool_id]
            except Exception as e:
                logging.debug(f"Error removing tool {tool_id} from memory: {e}")
                logging.debug(traceback.format_exc())
            # 4) Persist
            self._save_tools()
            logging.info(f"Removed tool: {getattr(tool, 'name', tool_id)} (ID: {tool_id})")
            return True
        except Exception as e:
            logging.debug(f"Error in remove_tool for {tool_id}: {e}")
            logging.debug(traceback.format_exc())
            return False

    def update_function_binding(self, tool_id: str, module_path: Optional[str] = None, refresh_signature: bool = True) -> bool:
        """Update function binding (module path and signature) for specified tool, return whether successful."""
        try:
            tool = self.tools.get(tool_id)
            if not tool:
                logging.info(f"Tool {tool_id} not found")
                return False
            if module_path:
                tool.module_path = module_path
            # Verify if callable exists
            func = self.resolve_callable(tool.function_name, tool.module_path)
            if func is None:
                logging.info(f"Function not found when updating binding: {tool.module_path}.{tool.function_name}")
                return False
            if refresh_signature:
                try:
                    sig = str(inspect.signature(func))
                    tool.function_signature = f"{tool.function_name}{sig}"
                    tool.function_signature_last_resolved_at = datetime.utcnow()
                except Exception as e:
                    logging.debug(f"Error refreshing signature for {tool.module_path}.{tool.function_name}: {e}")
                    logging.debug(traceback.format_exc())
            # Persist
            self._save_tools()
            return True
        except Exception as e:
            logging.debug(f"Error updating function binding for tool {tool_id}: {e}")
            logging.debug(traceback.format_exc())
            return False

    # def _generate_embedding(self, text: str) -> List[float]:
    #     """Generate text embedding (simplified version, should use real embedding model in practice)"""
    #     # This is a placeholder implementation, should integrate OpenAI embedding or other models in actual use
    #     # Using simple hash as demonstration here
    #     import hashlib

    #     hash_obj = hashlib.md5(text.encode())
    #     hash_hex = hash_obj.hexdigest()

    #     # Convert hex to 384-dimensional float vector (one of ChromaDB's default supported dimensions)
    #     embedding = []
    #     for i in range(0, len(hash_hex), 2):
    #         hex_pair = hash_hex[i:i+2]
    #         embedding.append(int(hex_pair, 16) / 255.0)  # Normalize to 0-1

    #     # Pad to 384 dimensions
    #     while len(embedding) < 384:
    #         embedding.extend(embedding[:min(len(embedding), 384 - len(embedding))])

    #     return embedding[:384]
    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate text embedding using OpenAI's 'text-embedding-3-large' model.
        API key and Base URL are defined directly in this function.

        **Warning:** Hardcoding sensitive information like keys directly in code is extremely unsafe in production environments.
        This implementation is only suitable for quick testing or personal projects.

        Args:
            text (str): Input text for generating embedding.

        Returns:
            List[float]: List of floats representing text embedding.

        Raises:
            Exception: If API call fails or other errors occur.
        """
        # 1. Define configuration directly inside function
        llm_config = get_openai_config()
        api_key = llm_config["api_key"]
        base_url = llm_config["base_url"]
       

        # 2. Create OpenAI client instance inside function
        # Note: Creating a new client every function call is less efficient than creating it during class initialization
        try:
            client = openai.OpenAI(api_key=api_key, base_url=base_url)
        except Exception as e:
            logging.debug(f"Error initializing OpenAI client: {e}")
            raise

        # 3. Prepare input and make API call
        if not text or not isinstance(text, str):
            logging.info("Warning: Input text is empty or invalid, returning empty list.")
            return []

        # OpenAI officially recommends replacing newlines to avoid potential issues
        processed_text = text.replace("\n", " ")

        try:
            response = client.embeddings.create(
                model="text-embedding-3-large",
                input=[processed_text]  # API requires a list as input
            )
            # 4. Extract and return embedding vector
            return response.data[0].embedding
        except openai.APIError as e:
            logging.debug(f"OpenAI API call failed: {e}")
            # Depending on specific situation, you might want to log error and return empty list or None
            # Here we choose to re-raise exception for caller to handle
            raise
        except Exception as e:
            logging.debug(f"Unknown error occurred while generating embedding: {e}")
            raise

    def recommend_tools_for_agent(self, task_description: str, previous_tool_ids: Optional[List[str]] = None,
                                 retrieval_k: int = 20, recommend_n: int = 5,
                                 semantic_threshold: float = None) -> List[Dict[str, Any]]:
        """Recommend high-quality candidate tool list with context information for Agent

        Args:
            task_description: Task description
            previous_tool_ids: List of previously executed tool IDs (cross-modal)
            retrieval_k: Number of tools for semantic recall
            recommend_n: Number of tools for final recommendation
            semantic_threshold: Semantic relevance threshold
        """
        try:
            if semantic_threshold is None:
                semantic_threshold = self.SEMANTIC_THRESHOLD

            # 1. Generate query vector
            query_embedding = self._generate_embedding(task_description)

            if len(self.tools) == 0:
                return []

            # 2. Semantic recall
            results = self.tool_collection.query(
                query_embeddings=[query_embedding],
                n_results=min(retrieval_k, len(self.tools))
            )

            if not results['ids'] or not results['ids'][0]:
                logging.info("No tools found in vector database")
                return []

            logging.info(f"[tool_recommendation]Found {len(results['ids'][0])} tools in vector database")

            # 3. Initial filtering: semantic relevance filtering
            candidates = []
            for i, tool_id in enumerate(results['ids'][0]):
                if tool_id not in self.tools:
                    continue

                tool = self.tools[tool_id]
                distance = results['distances'][0][i]
                similarity = 1 - distance  # Convert to similarity
                logging.info(f"[tool_recommendation]tool_name: {tool.name}, similarity: {similarity}")

                if similarity >= semantic_threshold:
                    candidates.append((tool, similarity))



            candidates.sort(key=lambda x: x[0].metrics.score, reverse=True)


            # 5. Build semantic recommendation results
            semantic_recommendations = []
            existing_tool_ids = set()

            for tool, similarity in candidates[:recommend_n]:
                context_hint = ""
                # Use historical parent tool information that was called before this tool
                parents = self.list_parents_with_names(tool.id, min_edge_weight=0.0)[:5]
                if parents:
                    try:
                        top = parents[0]
                        pname = top.get('name') or top.get('tool_id')
                        weight = top.get('weight', 0.0)
                        count = top.get('count', 0)
                        if count >= 3 and weight >= 0.6:
                            context_hint = f"Context Hint: Historically, '{pname}' often precedes this tool with {weight:.0%} success rate (based on {count} cases)."
                        else:
                            context_hint = f"Context Hint: This tool has often been used after '{pname}' with {weight:.0%} success rate."
                    except Exception:
                        pass

                semantic_recommendations.append({
                    "tool_id": tool.id,
                    "name": tool.name,
                    "description": tool.description,
                    "module_path": tool.module_path,
                    "function_name": tool.function_name,
                    "SBFL_quality_score": f"{tool.metrics.score:.2f}",
                    "SBFL_unified_score": f"{tool.metrics.unified_score:.2f}/1.0",
                    "semantic_similarity": f"{similarity:.2f}",
                    "context_hint": context_hint,
                    "data_source": tool.data_source,
                    "tags": tool.tags,
                    "success_rate": f"{tool.metrics.success_rate:.1%}",
                    "total_runs": tool.metrics.total_runs,
                    "is_successor": False,
                    "recommendation_type": "semantic"
                })
                existing_tool_ids.add(tool.id)

            # 6. Directly add successor tools of previous tools (not participating in sorting, directly added to final results)
            successor_recommendations = []
            if previous_tool_ids:
                for prev_tool_id in previous_tool_ids:
                    successors = self.tool_graph.get_successors(prev_tool_id, min_weight=0.5)
                    for successor_id, weight in successors.items():
                        if successor_id in self.tools and successor_id not in existing_tool_ids:
                            successor_tool = self.tools[successor_id]
                            prev_tool_name = self.tools.get(prev_tool_id, {}).name if prev_tool_id in self.tools else prev_tool_id
                            edge_data = self.tool_graph.graph[prev_tool_id][successor_id]
                            count = edge_data['count']

                            context_hint = f"Successor Tool: This tool often follows '{prev_tool_name}' with {weight:.0%} success rate (based on {count} historical cases). Highly recommended for your task."

                            successor_recommendations.append({
                                "tool_id": successor_tool.id,
                                "name": successor_tool.name,
                                "description": successor_tool.description,
                                "module_path": successor_tool.module_path,
                                "function_name": successor_tool.function_name,
                                "quality_score": f"{successor_tool.metrics.score:.2f}",
                                "SBFL unified_score": f"{tool.metrics.unified_score:.2f}/1.0",
                                "semantic_similarity": "Graph-based Recommendation",
                                "context_hint": context_hint,
                                "data_source": successor_tool.data_source,
                                "tags": successor_tool.tags,
                                "success_rate": f"{successor_tool.metrics.success_rate:.1%}",
                                "total_runs": successor_tool.metrics.total_runs,
                                "is_successor": True,
                                "recommendation_type": "successor",
                                "predecessor_tool": prev_tool_name,
                                "edge_weight": weight
                            })
                            existing_tool_ids.add(successor_id)
                            logging.info(f"[tool_recommendation]Added successor tool: {successor_tool.name} (follows {prev_tool_name})")

            # 7. Merge recommendation results: successor tools first, then semantic tools
            recommendations = successor_recommendations + semantic_recommendations

            # 8. If no recommendation results
            if not recommendations:
                logging.info(f"No tools meet semantic threshold {semantic_threshold},need TOOL GENERATION")
                return []

            return recommendations

        except Exception as e:
            logging.debug(f"Error in recommend_tools_for_agent: {e}")
            logging.debug(traceback.format_exc())
            return []

    def update_tool_after_execution(self, tool_id: str, execution_record: ToolExecutionRecord):
        """Update tool quality metrics after execution"""
        try:
            if tool_id not in self.tools:
                logging.info(f"Tool {tool_id} not found")
                return

            tool = self.tools[tool_id]
            metrics = tool.metrics


            # Update statistics (actual execution)
            metrics.total_runs += 1
            metrics.total_latency_ms += execution_record.latency_ms
            metrics.last_used_timestamp = execution_record.timestamp

            if execution_record.exit_code == 0:
                metrics.success_runs += 1
                metrics.last_success_timestamp = execution_record.timestamp

                # Only when execution is successful can it be considered effective
                if execution_record.is_effective:
                    metrics.effective_runs += 1

            if execution_record.rca_task_score is not None:
                metrics.accumulated_task_score += execution_record.rca_task_score

            # Add execution record
            tool.execution_history.append(execution_record)

            # Persist
            self._save_tools()

            logging.info(f"Updated tool {tool.name} metrics: score={metrics.score:.3f}, runs={metrics.total_runs}")

        except Exception as e:
            logging.debug(f"Error updating tool after execution: {e}")
            logging.debug(traceback.format_exc())

    def update_tool_after_execution_by_binding(self, module_path: str, function_name: str, execution_record: ToolExecutionRecord):
        """Update tool execution metrics based on module and function name, avoiding external ID lookup."""
        try:
            tool_id = None
            for tid, t in self.tools.items():
                if getattr(t, 'module_path', None) == module_path and getattr(t, 'function_name', None) == function_name:
                    tool_id = tid
                    break
            if not tool_id:
                logging.info(f"Tool not found for binding {module_path}.{function_name}")
                return
            self.update_tool_after_execution(tool_id, execution_record)
        except Exception as e:
            logging.debug(f"Error in update_tool_after_execution_by_binding for {module_path}.{function_name}: {e}")
            logging.debug(traceback.format_exc())

    def post_task_evolution(self, tool_chain: List[str], task_outcome: Dict[str, Any]):
        """Post-task evolution - update tool graph and end-to-end contribution"""
        try:
            task_success = task_outcome.get('task_overall_success', False)
            final_rca_score = task_outcome.get('final_rca_score', 0.0)

            # 1. Update tool graph (learn tool sequence relationships)
            for i in range(len(tool_chain) - 1):
                source_id = tool_chain[i]
                target_id = tool_chain[i + 1]
                self.tool_graph.add_or_update_edge(source_id, target_id, task_success)

            # 2. Only update tool's is_effective and rca_task_score when task succeeds
            if task_success:
                for tool_id in tool_chain:
                    if tool_id in self.tools:
                        # Update the last execution record's is_effective and rca_task_score
                        tool = self.tools[tool_id]
                        if tool.execution_history:
                            # Find the most recent non-chain contribution record
                            for record in reversed(tool.execution_history):
                                if not getattr(record, 'is_chain_contribution', False):
                                    # Only update when record hasn't been marked as effective yet
                                    if not record.is_effective:
                                        record.is_effective = True
                                        record.rca_task_score = final_rca_score

                                        # Only increase count when first marked as effective
                                        # Note: Don't modify metrics directly here, let update_tool_after_execution handle it
                                        # But since record is already in history, we need to manually update metrics
                                        if record.exit_code == 0:  # Only successful executions can be considered effective
                                            tool.metrics.effective_runs += 1
                                        tool.metrics.accumulated_task_score += final_rca_score

                                        logging.info(f"Updated tool {tool.name} effectiveness: is_effective=True, score={final_rca_score}")
                                    else:
                                        logging.info(f"Tool {tool.name} already marked as effective, skipping update")
                                    break
            else:
                logging.info(f"Task failed, skipping effectiveness updates for tool chain")

            # 3. Persist changes
            self._save_tools()

            try:
                display_chain = [f"{tid}({self.tools[tid].name})" if tid in self.tools else tid for tid in tool_chain]
            except Exception:
                display_chain = tool_chain
            logging.info(f"Post-task evolution completed for chain: {display_chain}")

        except Exception as e:
            logging.debug(f"Error in post_task_evolution: {e}")
            logging.debug(traceback.format_exc())

    def update_onp_for_unused_tools(self, used_tool_bindings: List[Tuple[str, str]], task_success: bool):
        """
        Update Onp count for all unused tools

        Args:
            used_tool_bindings: List of tool bindings used in this query [(module_path, function_name), ...]
            task_success: Whether this query succeeded
        """
        try:
            if not task_success:
                logging.info("[SBFL] Task failed, skipping Onp updates")
                return

            # Get all used tool IDs
            used_tool_ids = set()
            for module_path, function_name in used_tool_bindings:
                for tid, t in self.tools.items():
                    if getattr(t, 'module_path', None) == module_path and getattr(t, 'function_name', None) == function_name:
                        used_tool_ids.add(tid)
                        break

            # Update Onp count for all unused tools
            updated_count = 0
            for tid, tool in self.tools.items():
                if tid not in used_tool_ids:
                    tool.metrics.onp_count += 1
                    updated_count += 1
                    logging.info(f"[SBFL] Updated Onp for unused tool {tool.name}: onp_count={tool.metrics.onp_count}")

            logging.info(f"[SBFL] Updated Onp for {updated_count} unused tools")

            # Persist changes
            self._save_tools()

        except Exception as e:
            logging.debug(f"Error updating Onp for unused tools: {e}")
            logging.debug(traceback.format_exc())

    def post_task_evolution_by_bindings(self, tool_chain_bindings: List[Tuple[str, str]], task_outcome: Dict[str, Any]):
        """Execute post-task evolution based on (module_path, function_name) list."""
        try:
            logging.info(f"[tools debug]post_task_evolution_by_bindings: {tool_chain_bindings}")
            task_success = task_outcome.get('task_overall_success', False)
            final_rca_score = task_outcome.get('final_rca_score', 0.0)

            # Update Onp count for all unused tools (SBFL addition)
            self.update_onp_for_unused_tools(tool_chain_bindings, task_success)

            # Only update is_effective when task succeeds
            if task_success:
                for module_path, function_name in tool_chain_bindings:
                    for tid, t in self.tools.items():
                        if getattr(t, 'module_path', None) == module_path and getattr(t, 'function_name', None) == function_name:
                            # Update the last execution record's is_effective and rca_task_score
                            if t.execution_history:
                                # Find the most recent non-chain contribution record
                                for record in reversed(t.execution_history):
                                    if not getattr(record, 'is_chain_contribution', False):
                                        # Only update when record hasn't been marked as effective yet
                                        if not record.is_effective:
                                            record.is_effective = True
                                            record.rca_task_score = final_rca_score

                                            # Only increase count when first marked as effective
                                            # Only successful executions can be considered effective
                                            if record.exit_code == 0:
                                                t.metrics.effective_runs += 1
                                            t.metrics.accumulated_task_score += final_rca_score

                                            logging.info(f"Updated tool {t.name} effectiveness: is_effective=True, score={final_rca_score}")
                                        else:
                                            logging.info(f"Tool {t.name} already marked as effective, skipping update")
                                        break
                            break
            else:
                logging.info(f"Task failed, skipping effectiveness updates for tool bindings")

            # Update tool graph (learn tool sequence relationships)
            tool_ids: List[str] = []
            for module_path, function_name in tool_chain_bindings:
                for tid, t in self.tools.items():
                    if getattr(t, 'module_path', None) == module_path and getattr(t, 'function_name', None) == function_name:
                        tool_ids.append(tid)
                        break

            if tool_ids:
                # Update tool graph
                for i in range(len(tool_ids) - 1):
                    source_id = tool_ids[i]
                    target_id = tool_ids[i + 1]
                    self.tool_graph.add_or_update_edge(source_id, target_id, task_success)

                # Persist changes
                self._save_tools()

                try:
                    display_chain = [f"{tid}({self.tools[tid].name})" if tid in self.tools else tid for tid in tool_ids]
                except Exception:
                    display_chain = tool_ids
                logging.info(f"Post-task evolution completed for chain: {display_chain}")
            else:
                logging.info("No matching tools found for provided bindings; skipping evolution")

        except Exception as e:
            logging.debug(f"Error in post_task_evolution_by_bindings: {e}")
            logging.debug(traceback.format_exc())

    def get_tool_stats(self) -> Dict[str, Any]:
        """Get tool ecosystem statistics"""
        active_tools = [t for t in self.tools.values() if t.status == "active"]

        # SBFL related statistics
        total_oef = sum(t.metrics.oef for t in active_tools)
        total_oep = sum(t.metrics.oep for t in active_tools)
        total_onp = sum(t.metrics.onp for t in active_tools)

        # Calculate average Dstar2 score
        dstar2_scores = [t.metrics.dstar2_score for t in active_tools if t.metrics.total_runs > 0]
        avg_dstar2_score = sum(dstar2_scores) / len(dstar2_scores) if dstar2_scores else 0.0

        stats = {
            "total_tools": len(self.tools),
            "active_tools": len(active_tools),
            "graph_edges": self.tool_graph.graph.number_of_edges(),
            "graph_nodes": self.tool_graph.graph.number_of_nodes(),
            "top_quality_tools": [],
            "data_source_distribution": {},
            "total_executions": sum(t.metrics.total_runs for t in active_tools),
            "average_dstar2_score": round(avg_dstar2_score, 3),
            "sbfl_stats": {
                "total_oef": total_oef,
                "total_oep": total_oep,
                "total_onp": total_onp
            }
        }

        # Top 5 tools sorted by quality
        top_tools = sorted(active_tools, key=lambda x: x.metrics.score, reverse=True)[:5]
        for tool in top_tools:
            stats["top_quality_tools"].append({
                "name": tool.name,
                "score": round(tool.metrics.score, 3),
                "dstar2_score": round(tool.metrics.dstar2_score, 3),
                "runs": tool.metrics.total_runs,
                "success_rate": round(tool.metrics.success_rate, 3),
                "oef": tool.metrics.oef,
                "oep": tool.metrics.oep,
                "onp": tool.metrics.onp
            })

        # Data source distribution
        for tool in active_tools:
            source = tool.data_source or "unknown"
            stats["data_source_distribution"][source] = stats["data_source_distribution"].get(source, 0) + 1

        return stats

    def get_tools_overview(self, include_inactive: bool = False, min_edge_weight: float = 0.0) -> Dict[str, Any]:
        """Get overview of tools and graph, including summary of nodes and edges."""
        try:
            if include_inactive:
                tools_iter = list(self.tools.items())
            else:
                tools_iter = [(tid, t) for tid, t in self.tools.items() if getattr(t, 'status', 'active') == 'active']

            nodes = []
            for tid, t in tools_iter:
                nodes.append({
                    "id": tid,
                    "name": t.name,
                    "module_path": t.module_path,
                    "function_name": t.function_name,
                    "tags": t.tags,
                    "data_source": t.data_source,
                    "metrics": {
                        "runs": t.metrics.total_runs,
                        "success_runs": t.metrics.success_runs,
                        "success_rate": t.metrics.success_rate,
                        "effective_runs": t.metrics.effective_runs,
                        "avg_task_score": t.metrics.avg_task_score,
                        "score": t.metrics.score,
                        "dstar2_score": t.metrics.dstar2_score,
                        "oef": t.metrics.oef,
                        "oep": t.metrics.oep,
                        "onp": t.metrics.onp,
                    }
                })

            edges = []
            for u, v, data in self.tool_graph.graph.edges(data=True):
                weight = data.get('weight', 0.0)
                if weight < min_edge_weight:
                    continue
                # Only include edges where both source and target nodes exist in current tool set
                if u not in self.tools or v not in self.tools:
                    continue
                edges.append({
                    "source": u,
                    "target": v,
                    "weight": weight,
                    "count": data.get('count', 0),
                    "success_count": data.get('success_count', 0),
                    "source_name": self.tools[u].name,
                    "target_name": self.tools[v].name,
                })

            return {"nodes": nodes, "edges": edges}
        except Exception as e:
            logging.debug(f"Error building tools overview: {e}")
            logging.debug(traceback.format_exc())
            return {"nodes": [], "edges": []}

    def get_tool_details(self, tool_id: str, include_history: bool = False, history_limit: int = 20) -> Dict[str, Any]:
        """Get detailed information of a tool (optionally including history)."""
        try:
            t = self.tools.get(tool_id)
            if not t:
                return {"error": f"Tool {tool_id} not found"}
            details = {
                "id": tool_id,
                "name": t.name,
                "description": t.description,
                "module_path": t.module_path,
                "function_name": t.function_name,
                "function_signature": t.function_signature,
                "tags": t.tags,
                "data_source": t.data_source,
                "status": t.status,
                "created_at": t.created_at.isoformat(),
                "metrics": {
                    "runs": t.metrics.total_runs,
                    "success_runs": t.metrics.success_runs,
                    "success_rate": t.metrics.success_rate,
                    "effective_runs": t.metrics.effective_runs,
                    "avg_task_score": t.metrics.avg_task_score,
                    "total_latency_ms": t.metrics.total_latency_ms,
                    "last_used_timestamp": t.metrics.last_used_timestamp.isoformat() if t.metrics.last_used_timestamp else None,
                    "last_success_timestamp": t.metrics.last_success_timestamp.isoformat() if t.metrics.last_success_timestamp else None,
                    "score": t.metrics.score,
                }
            }
            if include_history:
                hist = t.execution_history[-history_limit:]
                details["history"] = [
                    {
                        "execution_id": r.execution_id,
                        "task_description": r.task_description,
                        "exit_code": r.exit_code,
                        "is_effective": r.is_effective,
                        "is_chain_contribution": getattr(r, 'is_chain_contribution', False),
                        "latency_ms": r.latency_ms,
                        "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                        "stdout_len": len(r.stdout or "") if isinstance(r.stdout, str) else None,
                        "stderr": r.stderr,
                    } for r in hist
                ]
            return details
        except Exception as e:
            logging.debug(f"Error getting tool details for {tool_id}: {e}")
            logging.debug(traceback.format_exc())
            return {"error": str(e)}

    def export_tool_graph_json(self, file_path: str, min_edge_weight: float = 0.0) -> bool:
        """Export current tool graph as JSON (including names), can filter edges by weight."""
        try:
            overview = self.get_tools_overview(include_inactive=True, min_edge_weight=min_edge_weight)
            payload = {
                "generated_at": datetime.utcnow().isoformat(),
                "nodes": overview.get("nodes", []),
                "edges": overview.get("edges", []),
            }
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            logging.info(f"Exported tool graph JSON to {file_path}")
            return True
        except Exception as e:
            logging.debug(f"Error exporting tool graph JSON to {file_path}: {e}")
            logging.debug(traceback.format_exc())
            return False


    def list_successors_with_names(self, tool_id: str, min_edge_weight: float = 0.5) -> List[Dict[str, Any]]:
        """List successor tools of a tool (including names, weights and statistics)."""
        try:
            if not self.tool_graph.graph.has_node(tool_id):
                return []
            results: List[Dict[str, Any]] = []
            for succ in self.tool_graph.graph.successors(tool_id):
                edge = self.tool_graph.graph[tool_id][succ]
                weight = edge.get('weight', 0.0)
                if weight < min_edge_weight:
                    continue
                results.append({
                    "tool_id": succ,
                    "name": self.tools[succ].name if succ in self.tools else succ,
                    "weight": weight,
                    "count": edge.get('count', 0),
                    "success_count": edge.get('success_count', 0)
                })
            results.sort(key=lambda x: x["weight"], reverse=True)
            return results
        except Exception as e:
            logging.debug(f"Error listing successors for {tool_id}: {e}")
            logging.debug(traceback.format_exc())
            return []

    def list_parents_with_names(self, tool_id: str, min_edge_weight: float = 0.5) -> List[Dict[str, Any]]:
        """List parent tools of a tool (including names, weights and statistics)."""
        try:
            if not self.tool_graph.graph.has_node(tool_id):
                return []
            results: List[Dict[str, Any]] = []
            for parent in self.tool_graph.graph.predecessors(tool_id):
                edge = self.tool_graph.graph[parent][tool_id]
                weight = edge.get('weight', 0.0)
                if weight < min_edge_weight:
                    continue
                results.append({
                    "tool_id": parent,
                    "name": self.tools[parent].name if parent in self.tools else parent,
                    "weight": weight,
                    "count": edge.get('count', 0),
                    "success_count": edge.get('success_count', 0)
                })
            results.sort(key=lambda x: x["weight"], reverse=True)
            return results
        except Exception as e:
            logging.debug(f"Error listing parents for {tool_id}: {e}")
            logging.debug(traceback.format_exc())
            return []

# --- Sample code for testing ---
if __name__ == "__main__":
    # Instantiate the class containing this method
    my_instance = ToolEcosystemManager()

    # Define sample text
    sample_text = "This is an example of calling OpenAI API directly within a function."

    logging.info(f"Generating embedding for text: '{sample_text}'")

    try:
        # Call the method
        # Note: Please ensure you have filled in your own valid key in the api_key variable in the code above
        embedding_vector = my_instance._generate_embedding(sample_text)

        # Print results for verification
        if embedding_vector:
            logging.info("\n--- Results ---")
            logging.info(f"Successfully generated embedding!")
            logging.info(f"Vector dimension: {len(embedding_vector)}")
            logging.info(f"First 5 values of vector: {embedding_vector[:5]}")
    except Exception as e:
        logging.debug(f"\nProgram execution error: {e}")
        logging.debug("Please check if your API key is correct, valid, and your account has sufficient balance.")

    # Store in vector database
    try:
        my_instance.tool_collection.add(
            embeddings=[embedding_vector],
            documents=["sample_text"],
            metadatas=[{
                "tool_id": 77777,
            }],
            ids=["1"]
        )
    except Exception as e:
        logging.debug(traceback.format_exc())
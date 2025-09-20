import asyncio
from workflow import DiagnosisWorkflow
import pprint
import re
import json
from config_manager import get_data_paths
import time
import signal
import argparse  # Add argparse import
from eval.evaluate import evaluate
from agents import (
    get_metric_tool_ecosystem_manager,
    get_log_tool_ecosystem_manager,
    get_trace_tool_ecosystem_manager,
    get_metric_tool_chain,
    get_log_tool_chain,
    get_trace_tool_chain,
)


# Add a timeout handler decorator
def timeout_handler(signum, frame):
    raise TimeoutError("Function execution timeout (exceeded 30 minutes)")


def is_tool_execution_result_valid(execution_result):
    """
    check if tool execution result is valid (non-empty)

    Args:
        execution_result: tool execution result

    Returns:
        bool: True if result is valid (non-empty), False otherwise
    """
    if execution_result is None:
        return False

    if isinstance(execution_result, str):
        # check if string is empty, contains only whitespace characters, or is one of the special values
        stripped = execution_result.strip()
        return bool(stripped) and stripped not in ["[]", "{}", "None", "null", "anomaly_events=[]",]

    if isinstance(execution_result, list):
        # check if list is empty
        return len(execution_result) > 0

    if isinstance(execution_result, dict):
        # check if dict is empty
        return len(execution_result) > 0

    # for other types, convert to string and check
    try:
        str_result = str(execution_result).strip()
        # check if empty string, "[]", "{}", "None" etc
        return str_result and str_result not in ["[]", "{}", "None", "null"]
    except Exception:
        return False


def persist_tools_based_on_scoring(final_result, generated_tools, scoring_points):
    """
    decide whether to persist tools based on scoring result and tool execution result

    Args:
        final_result: final diagnosis result
        generated_tools: list of generated tool info
        scoring_points: scoring criteria
    """
    print(f"[Tool Persistence] Evaluating {len(generated_tools)} generated tools for persistence...")

    if not generated_tools:
        print(f"[Tool Persistence] No tools generated, nothing to persist")
        return

    if scoring_points is None:
        print(f"[Tool Persistence] No scoring points provided, checking execution results only")
        # no scoring info, only check execution results
        for tool_info in generated_tools:
            execution_result = tool_info.get('execution_result', '')
            coder_name = tool_info.get('coder_name', 'unknown')

            if is_tool_execution_result_valid(execution_result):
                print(f"[Tool Persistence] Tool {coder_name} has valid execution result, persisting")
                persist_single_tool(tool_info)
            else:
                print(f"[Tool Persistence] Tool {coder_name} has empty/invalid execution result, skipping persistence")
                print(f"[Tool Persistence] Execution result: {repr(execution_result)}")
        return

    try:
        from eval.evaluate import evaluate
        from code_utils import save_code_blocks, save_code_as_functions

        passing_criteria, _, bin_score, _, _, _, _ = evaluate(final_result, scoring_points)

        if bin_score > 0:
            print(f"[Tool Persistence] Task succeeded (score: {bin_score}), checking execution results for each tool")
            # task succeeded, but still need to check each tool's execution results
            for tool_info in generated_tools:
                execution_result = tool_info.get('execution_result', '')
                coder_name = tool_info.get('coder_name', 'unknown')

                if is_tool_execution_result_valid(execution_result):
                    print(f"[Tool Persistence] Tool {coder_name} has valid execution result, persisting")
                    persist_single_tool(tool_info)
                else:
                    print(f"[Tool Persistence] Tool {coder_name} has empty/invalid execution result, skipping persistence")
                    print(f"[Tool Persistence] Execution result: {repr(execution_result)}")
        else:
            print(f"[Tool Persistence] Task failed (score: {bin_score}), skipping tool persistence")
            # task failed, do not persist tools
            for tool_info in generated_tools:
                print(f"[Tool Persistence] Skipping persistence for {tool_info.get('coder_name', 'unknown')} tool due to low task score")

    except Exception as e:
        print(f"[Tool Persistence] Error during scoring evaluation: {e}")
        import traceback
        print(traceback.format_exc())
        print(f"[Tool Persistence] Falling back to checking execution results only")
        # error, check execution results only
        for tool_info in generated_tools:
            execution_result = tool_info.get('execution_result', '')
            coder_name = tool_info.get('coder_name', 'unknown')

            if is_tool_execution_result_valid(execution_result):
                print(f"[Tool Persistence] Tool {coder_name} has valid execution result, persisting (fallback)")
                persist_single_tool(tool_info)
            else:
                print(f"[Tool Persistence] Tool {coder_name} has empty/invalid execution result, skipping persistence (fallback)")
                print(f"[Tool Persistence] Execution result: {repr(execution_result)}")


def persist_single_tool(tool_info):
    """
    persist single tool

    Args:
        tool_info: dict containing blocks_to_save, task_description, output_file etc
    """
    try:
        from code_utils import save_code_blocks, save_code_as_functions

        blocks_to_save = tool_info.get('blocks_to_save', [])
        task_description = tool_info.get('task_description', '')
        output_file = tool_info.get('output_file', 'tools/default_tools.py')
        coder_name = tool_info.get('coder_name', 'unknown')

        if not blocks_to_save:
            print(f"[Tool Persistence] No code blocks to save for {coder_name}")
            return

        # save code blocks
        save_code_blocks(blocks_to_save)

        data_paths = get_data_paths()
        base_path = data_paths.get("base_path", "./coding/dataset")

        # save as functions, enable path conversion
        save_code_as_functions(
            blocks_to_save,
            task_description,
            output_file=output_file,
            convert_paths=True,
            base_path=base_path
        )

        # sync new tools to tool ecosystem, pass task description
        try:
            from agents import get_metric_tool_ecosystem_manager, get_log_tool_ecosystem_manager, get_trace_tool_ecosystem_manager

            # select manager based on coder type - use config manager to get correct module path
            from config_manager import get_tool_files_with_env_override
            tool_files = get_tool_files_with_env_override()

            if 'metric' in coder_name:
                manager = get_metric_tool_ecosystem_manager()
                module_path = tool_files["metric_tools"]
                data_source = "metric"
            elif 'log' in coder_name:
                manager = get_log_tool_ecosystem_manager()
                module_path = tool_files["log_tools"]
                data_source = "log"
            elif 'trace' in coder_name:
                manager = get_trace_tool_ecosystem_manager()
                module_path = tool_files["trace_tools"]
                data_source = "trace"
            else:
                print(f"[Tool Persistence] Unknown coder type: {coder_name}, skipping ecosystem sync")
                return

            # resync module, this time including task description
            print(f"[Tool Persistence] Syncing {module_path} with task context to ecosystem...")
            manager.sync_tools_from_module_with_task_context(module_path, task_description,
                                                           tags=[data_source, "generated"],
                                                           data_source=data_source,
                                                           author="coder")

        except Exception as e:
            print(f"[Tool Persistence] Error syncing to ecosystem: {e}")
            import traceback
            print(traceback.format_exc())

        print(f"[Tool Persistence] Successfully persisted {coder_name} tool to {output_file}")

    except Exception as e:
        print(f"[Tool Persistence] Error persisting tool {tool_info.get('coder_name', 'unknown')}: {e}")
        import traceback
        print(traceback.format_exc())

async def opslen_rca(instruction=None, dataset=None, record_idx=None, model=None, scoring_points=None, enable_log_redirect=False):
    """
    RCA (Root Cause Analysis) entry function for the operations diagnosis system

    Args:
        instruction: User-provided diagnosis instruction, if None then use default instruction
        dataset: Dataset name, used for generating output filename
        record_idx: Record index, used for generating output filename
        model: Model name, used for log redirection
        scoring_points: Scoring points for evaluation
        enable_log_redirect: Whether to enable log redirection to file

    Returns:
        tuple: (Root cause analysis result, task_completion_types, tool_library_count)
    """
    # Initialize variables to prevent undefined variable access in exception handling
    final_result = None
    task_completion_types = []
    tool_library_count = 0

    try:
        # Set 1-hour timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(3600)  
        
        # Direct output to terminal without capturing
        try:
            async with await DiagnosisWorkflow.create() as workflow:
                # If no instruction is provided, use default instruction
                if instruction is None:
                    user_query = "On March 10, 2021, between 15:00 and 15:30, two system failures were encountered. The components responsible for these failures and the reasons behind them are not yet known. Please identify the root cause components and the root cause reasons."
                else:
                    user_query = instruction
                    
                queried_issue = ""
                reference_books = [""]
                
                print(f"[Execution Start] Processing instruction: {user_query}")
                print(f"[Dataset: {dataset}, Record Index: {record_idx}]")
                print(f"[Timeout limit set: 30 minutes]")
                
                try:
                    start_time = time.time()
                    diagnosis_result, generated_tools = await workflow.run_diagnosis(
                        user_query=user_query,
                        queried_issue=queried_issue,
                        reference_books=reference_books,
                        scoring_points=scoring_points
                    )
                    end_time = time.time()
                    print(f"[Execution Complete] Time taken: {end_time - start_time:.2f} seconds")

                    print("[Main] Diagnosis Result:")
                    pprint.pprint(diagnosis_result)

                    final_result = diagnosis_result["root_cause"]
                    task_completion_types = diagnosis_result.get("task_completion_types", [])
                    tool_library_count = diagnosis_result.get("tool_library_count", 0)

                    print(f"[Statistics] Task completion types: {task_completion_types}")

                    if "```json" in final_result:
                        final_result = re.search(r"```json\n(.*)\n```", final_result, re.S).group(1).strip()

                    print("--------------------------------Final Result--------------------------------")
                    print(final_result)
                    print("--------------------------------Final Result--------------------------------")

                    # persist tools based on scoring
                    persist_tools_based_on_scoring(final_result, generated_tools, scoring_points)
                except Exception as inner_e:
                    print(f"[ERROR] Error during diagnosis execution: {str(inner_e)}")
                    print(f"[ERROR] Error type: {type(inner_e).__name__}")
                    import traceback
                    print(f"[ERROR] Full traceback:")
                    traceback.print_exc()
                    final_result = None
                    task_completion_types = []
                    tool_library_count = 0
        except Exception as workflow_e:
            print(f"[ERROR] Error during workflow creation: {str(workflow_e)}")
            print(f"[ERROR] Error type: {type(workflow_e).__name__}")
            import traceback
            traceback.print_exc()
            final_result = None
            task_completion_types = []
            tool_library_count = 0
        
        # Disable timeout alarm
        signal.alarm(0)
        
        
        # If final_result is None (possibly due to ignored exceptions during code execution), use static prediction
        if final_result is None:
            print("--------------------------------final_result is empty--------------------------------")
            final_result = get_static_prediction()
            task_completion_types = []
            tool_library_count = 0
            print("--------------------------------final_result is empty--------------------------------")
        else:
            # Post-task evolution if scoring_points is provided
            try:
                if scoring_points:
                    # Evaluate to get bin_score and success flag
                    passing_criteria, _, bin_score, _, _, _, _ = evaluate(final_result, scoring_points)
                    task_overall_success = bool(passing_criteria) and bin_score > 0
                    task_outcome = {
                        'task_overall_success': task_overall_success,
                        'final_rca_score': float(bin_score)
                    }
                    print(f"[Post-task Evolution] Task success: {task_overall_success}, Score: {bin_score}")

                    # Always perform post-task evolution for SBFL (Onp updates), regardless of task success
                    # Post evolve per domain
                    try:
                        metric_chain = get_metric_tool_chain()
                        get_metric_tool_ecosystem_manager().post_task_evolution_by_bindings(metric_chain, task_outcome)
                        print(f"[Post-task Evolution] Metric tools evolved with chain: {metric_chain}")
                    except Exception as e:
                        print(f"[WARN] Metric post_task_evolution failed: {e}")
                    try:
                        log_chain = get_log_tool_chain()
                        get_log_tool_ecosystem_manager().post_task_evolution_by_bindings(log_chain, task_outcome)
                        print(f"[Post-task Evolution] Log tools evolved with chain: {log_chain}")
                    except Exception as e:
                        print(f"[WARN] Log post_task_evolution failed: {e}")
                    try:
                        trace_chain = get_trace_tool_chain()
                        get_trace_tool_ecosystem_manager().post_task_evolution_by_bindings(trace_chain, task_outcome)
                        print(f"[Post-task Evolution] Trace tools evolved with chain: {trace_chain}")
                    except Exception as e:
                        print(f"[WARN] Trace post_task_evolution failed: {e}")

                    try:
                        from agents import get_unified_tool_chain
                        unified_chain = get_unified_tool_chain()
                        if unified_chain:
                            get_metric_tool_ecosystem_manager().post_task_evolution(unified_chain, task_outcome)

                            # get tool names
                            try:
                                tool_names = []
                                for tool_id in unified_chain:
                                    if tool_id in get_metric_tool_ecosystem_manager().tools:
                                        tool = get_metric_tool_ecosystem_manager().tools[tool_id]
                                        tool_name = tool.name if tool.name else tool.function_name
                                        tool_names.append(f"{tool_name}({tool_id})")
                                    elif tool_id in get_log_tool_ecosystem_manager().tools: 
                                        tool = get_log_tool_ecosystem_manager().tools[tool_id]
                                        tool_name = tool.name if tool.name else tool.function_name
                                        tool_names.append(f"{tool_name}({tool_id})")
                                    elif tool_id in get_trace_tool_ecosystem_manager().tools: 
                                        tool = get_trace_tool_ecosystem_manager().tools[tool_id]
                                        tool_name = tool.name if tool.name else tool.function_name
                                        tool_names.append(f"{tool_name}({tool_id})")
                                    else:
                                        tool_names.append(tool_id)
                                print(f"[Post-task Evolution] Cross-modal unified chain evolved: {tool_names}")
                            except Exception as name_e:
                                print(f"[Post-task Evolution] Cross-modal unified chain evolved: {unified_chain}")
                                print(f"[WARN] Failed to get tool names: {name_e}")
                        else:
                            print(f"[Post-task Evolution] No unified tool chain to evolve")
                    except Exception as e:
                        print(f"[WARN] Unified chain post_task_evolution failed: {e}")
                        import traceback
                        print(traceback.format_exc())

                    if not task_overall_success:
                        print(f"[Post-task Evolution] Task failed (score: {bin_score}), but Onp updates were still performed for SBFL")
            except Exception as e:
                import traceback
                print(f"[WARN] post_task_evolution failed: {e}")
                print(traceback.format_exc())
        return final_result, task_completion_types, tool_library_count
            
    except TimeoutError as te:
        print(f"Diagnosis process timeout: {str(te)}")
        # Return a timeout prediction result
        static_result = get_static_prediction()
        print(f"Returning static prediction result: {static_result}")
        return static_result, [], 0
    except Exception as e:
        # Ensure timeout alarm is disabled
        signal.alarm(0)

        print(f"[ERROR] Error during diagnosis process: {str(e)}")
        print(f"[ERROR] Error type: {type(e).__name__}")
        import traceback
        print(f"[ERROR] Full traceback:")
        traceback.print_exc()
        # Return a static prediction result
        static_result = get_static_prediction()
        print(f"[FALLBACK] Returning static prediction result: {static_result}")
        return static_result, [], 0

def get_static_prediction():
    """Return a static prediction result for testing the evaluation system"""
    return json.dumps({
        "root_cause_component": "NA",
        "root_cause_reason": "NA",
        "root_cause_time": "NA"
    }, ensure_ascii=False)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='OpsLens RCA System')
    parser.add_argument('--query', type=str, help='The RCA query to process')
    parser.add_argument('--scoring_points', type=str, help='The scoring points for post-task evolution')
    args = parser.parse_args()


    print("--------------------------------Execution Start--------------------------------")
    # Use the query from command line if provided, otherwise use default
    query = args.query if args.query else "On March 10, 2021, between 15:00 and 15:30, two system failures were encountered. The components responsible for these failures and the reasons behind them are not yet known. Please identify the root cause components and the root cause reasons."
    gt = args.scoring_points if args.scoring_points else None

    final_result, task_completion_types, tool_library_count = asyncio.run(opslen_rca(instruction=query, scoring_points=gt))
    print("--------------------------------Final Result--------------------------------")
    print(final_result)
    print(f"Task completion types: {task_completion_types}")
    print("--------------------------------Final Result--------------------------------")

    
    
    
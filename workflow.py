from typing import Dict, List, Optional
from agents import *
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken, SingleThreadedAgentRuntime
import pprint
from autogen_core import AgentId
from docker_code_executor import DockerCommandLineCodeExecutor
from coder import *
import json
from datetime import datetime, timedelta
import logging
from coder import MetricCoder, LogCoder, TraceCoder

from prompt import get_prompt_module
from config_manager import get_prompt_type_with_env_override, get_data_paths
from prompt.WorkflowPrompt import *

prompt_type = get_prompt_type_with_env_override()
prompt_module = get_prompt_module(prompt_type)

background = getattr(prompt_module, "background", "")
data_description = getattr(prompt_module, "data_description", {})
diagnosis_plan = getattr(prompt_module, "diagnosis_plan", "")


class DiagnosisWorkflow:
    def __init__(self):
        self.agents = {
            "planner": planner_agent,
            "investigator": investigator_agent,
            "metric_explorer": metric_explorer,
            "log_explorer": log_explorer,
            "trace_explorer": trace_explorer,
            "reasoner": reasoner_agent,
        }
        

        self.model_client = model_client
        self.runtime = SingleThreadedAgentRuntime()
        
        
        self.metric_coder = None
        self.log_coder = None
        self.trace_coder = None
        
        self.executor_agent = None
        self.docker_executor = None

        
        self.explorer_notebook = NotebookSystem()
        self.coder_notebook = NotebookSystem()
        
        self.timing = {
            "plan": None,
            "investigate": None,
            "coder": timedelta(),
            "reason": None,
            "total": None
        }
        
        self.llm_call_count = {
            "planner": 0,
            "investigator": 0,
            "metric_explorer": 0,
            "log_explorer": 0,
            "trace_explorer": 0,
            "reasoner": 0,
            "metric_coder": 0,
            "log_coder": 0,
            "trace_coder": 0,
            "total": 0
        }
        
        self.token_usage = {
            "planner": {"prompt": 0, "completion": 0, "total": 0},
            "investigator": {"prompt": 0, "completion": 0, "total": 0},
            "metric_explorer": {"prompt": 0, "completion": 0, "total": 0},
            "log_explorer": {"prompt": 0, "completion": 0, "total": 0},
            "trace_explorer": {"prompt": 0, "completion": 0, "total": 0},
            "reasoner": {"prompt": 0, "completion": 0, "total": 0},
            "metric_coder": {"prompt": 0, "completion": 0, "total": 0},
            "log_coder": {"prompt": 0, "completion": 0, "total": 0},
            "trace_coder": {"prompt": 0, "completion": 0, "total": 0},
            "total": {"prompt": 0, "completion": 0, "total": 0}
        }

        self.task_completion_types = []  # record the completion type of each task, 'g' for generate_tool, 'p' for prompt
        self.tool_library_count = 0  # record the number of tools in the tool library

    @classmethod
    async def create(cls):
        """Asynchronous factory method to create and initialize DiagnosisWorkflow instance"""
        workflow = cls()
        
        # 1. Create and start executor first
        data_paths = get_data_paths()
        work_dir = data_paths.get("base_path", "./coding/dataset")
        workflow.docker_executor = DockerCommandLineCodeExecutor(work_dir=work_dir, auto_remove=False, container_name='opslens_fse')
        await workflow.docker_executor.start()
        workflow.executor_agent = await Executor.register(
            workflow.runtime, 
            "executor", 
            lambda: Executor(workflow.docker_executor)
        )
        
        # 2. Create coders
        workflow.metric_coder = await MetricCoder.register(
            workflow.runtime,
            "metric_coder",
            lambda: MetricCoder(
                reason_model_client,
                name="metric_coder"
            ),
        )
        workflow.log_coder = await LogCoder.register(
            workflow.runtime,
            "log_coder",
            lambda: LogCoder(
                reason_model_client,
                name="log_coder"
            ),
        )
        workflow.trace_coder = await TraceCoder.register(
            workflow.runtime,
            "trace_coder",
            lambda: TraceCoder(
                reason_model_client,
                name="trace_coder"
            ),
        )
        
        # 3. Update agents dictionary
        workflow.agents.update({
            "executor": workflow.executor_agent,
            "metric_coder": workflow.metric_coder,
            "log_coder": workflow.log_coder,
            "trace_coder": workflow.trace_coder
        })
        
        # 4. Start runtime last
        workflow.runtime.start()
        
        # 5. Check and update LLM call count after completion
        if hasattr(MetricCoder, 'get_llm_call_count'):
            workflow.llm_call_count["metric_coder"] = MetricCoder.get_llm_call_count()
            workflow.llm_call_count["total"] += MetricCoder.get_llm_call_count()
        
        if hasattr(LogCoder, 'get_llm_call_count'):  
            workflow.llm_call_count["log_coder"] = LogCoder.get_llm_call_count()
            workflow.llm_call_count["total"] += LogCoder.get_llm_call_count()
            
        if hasattr(TraceCoder, 'get_llm_call_count'):
            workflow.llm_call_count["trace_coder"] = TraceCoder.get_llm_call_count()
            workflow.llm_call_count["total"] += TraceCoder.get_llm_call_count()
        
        return workflow

    async def cleanup(self):
        """Clean up resources"""
        if self.docker_executor:
            await self.docker_executor.stop()
            
        # Update LLM call count statistics
        self.llm_call_count["metric_coder"] = MetricCoder.get_llm_call_count()
        self.llm_call_count["log_coder"] = LogCoder.get_llm_call_count()
        self.llm_call_count["trace_coder"] = TraceCoder.get_llm_call_count()
        self.llm_call_count["total"] = self.llm_call_count["metric_coder"] + self.llm_call_count["log_coder"] + self.llm_call_count["trace_coder"]
        
        # Update token usage statistics
        if hasattr(MetricCoder, 'get_token_usage'):
            metric_usage = MetricCoder.get_token_usage()
            self.token_usage["metric_coder"]["prompt"] += metric_usage["prompt"]
            self.token_usage["metric_coder"]["completion"] += metric_usage["completion"]
            self.token_usage["metric_coder"]["total"] += metric_usage["total"]
            
            self.token_usage["total"]["prompt"] += metric_usage["prompt"]
            self.token_usage["total"]["completion"] += metric_usage["completion"]
            self.token_usage["total"]["total"] += metric_usage["total"]
            
        if hasattr(LogCoder, 'get_token_usage'):
            log_usage = LogCoder.get_token_usage()
            self.token_usage["log_coder"]["prompt"] += log_usage["prompt"]
            self.token_usage["log_coder"]["completion"] += log_usage["completion"]
            self.token_usage["log_coder"]["total"] += log_usage["total"]
            
            self.token_usage["total"]["prompt"] += log_usage["prompt"]
            self.token_usage["total"]["completion"] += log_usage["completion"]
            self.token_usage["total"]["total"] += log_usage["total"]
            
        if hasattr(TraceCoder, 'get_token_usage'):
            trace_usage = TraceCoder.get_token_usage()
            self.token_usage["trace_coder"]["prompt"] += trace_usage["prompt"]
            self.token_usage["trace_coder"]["completion"] += trace_usage["completion"]
            self.token_usage["trace_coder"]["total"] += trace_usage["total"]
            
            self.token_usage["total"]["prompt"] += trace_usage["prompt"]
            self.token_usage["total"]["completion"] += trace_usage["completion"]
            self.token_usage["total"]["total"] += trace_usage["total"]
        
        # Output LLM call count statistics
        logging.info(f"\n{'='*50}")
        logging.info(f"[LLM Call Statistics Summary]")
        for agent, count in self.llm_call_count.items():
            if count > 0 and agent != "total":
                logging.info(f"  - {agent}: {count}")
        logging.info(f"  - Total: {self.llm_call_count['total']}")
        
        # Output Token usage statistics
        logging.info(f"\n[Token Usage Statistics Summary]")
        for agent, usage in self.token_usage.items():
            if usage["total"] > 0 and agent != "total":
                logging.info(f"  - {agent}: input={usage['prompt']}, output={usage['completion']}, total={usage['total']}")
        logging.info(f"  - Total: input={self.token_usage['total']['prompt']}, output={self.token_usage['total']['completion']}, total={self.token_usage['total']['total']}")

        
        # Output time usage statistics
        logging.info(f"[Time Statistics] Diagnosis process end, total time: {self.timing['total']}")
        logging.info(f"[Time Statistics] Time usage by phase:")
        logging.info(f"  - Planning phase: {self.timing['plan']}")
        logging.info(f"  - Investigation phase: {self.timing['investigate']}")
        logging.info(f"    - Coder part: {self.timing['coder']}")
        logging.info(f"  - Reasoning phase: {self.timing['reason']}")

        # Output task_completion_type & tool_library_count
        logging.info(f"[task_completion_type] {self.task_completion_types}")
        logging.info(f"[tool_library_count] {self.tool_library_count}")
        
        
        logging.info(f"{'='*50}\n")

    async def __aenter__(self):
        """Async context manager entry point"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit point"""
        await self.cleanup()

    async def generate_tool(self, task_description: str, explorer_name: str, scoring_points: str = None) -> tuple:
        start_time = datetime.now()
        try:
            self.runtime.start()
        except Exception as e:
            print(f"Runtime start warning: {e}")
        
        coder_name = explorer_name[:-8] + "coder"
        
       
        # Add coder's identifier when sending messages,notice that telecom dataset without log_coder
        message_content = f"""
        [From {coder_name}]
        {data_description[f"{coder_name}"]}
        <task>{task_description}</task>
        """
        
        
        
        self.coder_notebook.save_task(coder_name, task_description)
        enriched_message = self.coder_notebook.enrich_message(message_content, coder_name)
        
        try:
            await self.runtime.send_message(
                message=Message(enriched_message),
                recipient=AgentId(f"{coder_name}", "default")
            )
            
            await self.runtime.stop_when_idle()
            
            # Get execution result
            code_blocks = Executor.get_execution_code()
            if not code_blocks:
                print(f"Warning: No code blocks generated for {coder_name}")
                return None
            
            # Prepare list of code blocks to save
            blocks_to_save = []
            for i, code_block in enumerate(code_blocks, 1):
                # Process only Python code blocks, skip other formats (e.g., JSON)
                if code_block.language.lower() in ['python', 'py']:
                    logging.info(f"\nCode block {i}:")
                    logging.info(f"Language: {code_block.language}")
                    blocks_to_save.append(code_block)
                else:
                    logging.info(f"Skipping non-Python code block, language: {code_block.language}")
            
            execution_result = Executor.get_execution_result()
            logging.info("======generate_tool execution_result=======")
            pprint.pprint(execution_result)
            logging.info("======generate_tool execution_result=======")
            self.coder_notebook.save_response(coder_name, execution_result)

            # Update coder time
            end_time = datetime.now()
            self.timing["coder"] += end_time - start_time
            logging.info(f"[Time Statistics] {coder_name} tool generation time: {end_time - start_time}")

            # get tool file path
            from config_manager import get_tool_files_with_env_override
            tool_files = get_tool_files_with_env_override()

            coder_type = coder_name.split('_')[0]  # metric, log, trace
            tool_file_key = f"{coder_type}_tools"
            output_file_module = tool_files.get(tool_file_key, f"tools.{coder_type}_tools")
            output_file = output_file_module.replace('.', '/') + '.py'

            tool_info = {
                'blocks_to_save': blocks_to_save if blocks_to_save else [],
                'task_description': task_description,
                'coder_name': coder_name,
                'output_file': output_file,
                'execution_result': execution_result  # added execution result
            }

            return execution_result, tool_info
        
        except Exception as e:
            # Update coder time (even if error occurs)
            end_time = datetime.now()
            self.timing["coder"] += end_time - start_time
            logging.info(f"[Time Statistics] {coder_name} tool generation time (error): {end_time - start_time}")

            print(f"Error in generate_tool for {coder_name}: {str(e)}")
            print(f"Error details: {type(e).__name__}")  # Add more detailed error information
            import traceback
            print(traceback.format_exc())

            from config_manager import get_tool_files_with_env_override
            tool_files = get_tool_files_with_env_override()

            coder_type = coder_name.split('_')[0]  # metric, log, trace
            tool_file_key = f"{coder_type}_tools"
            output_file_module = tool_files.get(tool_file_key, f"tools.{coder_type}_tools")
            output_file = output_file_module.replace('.', '/') + '.py'

            return None, {'blocks_to_save': [], 'task_description': task_description, 'coder_name': coder_name, 'output_file': output_file}


    def _initialize_tool_library_count(self):
        """Initialize and record the number of tools in the tool library"""
        try:
            from agents import get_metric_tool_ecosystem_manager, get_log_tool_ecosystem_manager, get_trace_tool_ecosystem_manager
            metric_manager = get_metric_tool_ecosystem_manager()
            log_manager = get_log_tool_ecosystem_manager()
            trace_manager = get_trace_tool_ecosystem_manager()
            self.tool_library_count = len(metric_manager.tools) + len(log_manager.tools) + len(trace_manager.tools)
            logging.info(f"[Tool Library Count] Total tools in library: {self.tool_library_count}")
        except Exception as e:
            logging.info(f"[Tool Library Count] Error getting tool count: {e}")
            self.tool_library_count = 0

    async def _get_investigator_decision(self, investigator_msg: str, available_explorers: dict) -> tuple:
        """Get and process investigator's decision

        Returns:
            tuple: (explorer_name, task) if successful, (None, None) if investigation should end
        """
        investigator_prompt = investigator_prompt_template.format(
            investigator_msg=investigator_msg,
            available_explorers=available_explorers
        )

        response = await self.agents["investigator"].on_messages(
            [TextMessage(content=investigator_prompt, source="user")],
            cancellation_token=CancellationToken(),
        )

        # Update statistics
        self.llm_call_count["investigator"] += 1
        self.llm_call_count["total"] += 1

        prompt_tokens, completion_tokens = self._extract_token_usage(response, "investigator", "main call")
        self.token_usage["investigator"]["prompt"] += prompt_tokens
        self.token_usage["investigator"]["completion"] += completion_tokens
        self.token_usage["investigator"]["total"] += prompt_tokens + completion_tokens
        self.token_usage["total"]["prompt"] += prompt_tokens
        self.token_usage["total"]["completion"] += completion_tokens
        self.token_usage["total"]["total"] += prompt_tokens + completion_tokens


        await self.print_llm_response("investigator", response, system_message=investigator_prompt)
        decision = response.chat_message.content

        if "INVESTIGATION_COMPLETE" in decision:
            return None, None

        return self._parse_investigator_decision(decision, available_explorers)

    def _parse_investigator_decision(self, decision: str, available_explorers: dict) -> tuple:
        """Parse investigator's decision with enhanced robustness

        Returns:
            tuple: (explorer_name, task) if successful
        """
        try:
            # Enhanced text cleaning and preprocessing
            decision = decision.strip()

            # Replace problematic characters that can break JSON parsing
            decision = decision.replace('—', '-').replace('–', '-')
            decision = decision.replace('"', '"').replace('"', '"')
            decision = decision.replace(''', "'").replace(''', "'")

            import re
            decision = re.sub(r'[^\x20-\x7E\n\t]', '', decision)

            # Handle markdown code block format
            if decision.startswith("```json") and decision.endswith("```"):
                decision = decision[7:-3].strip()
            elif decision.startswith("```") and decision.endswith("```"):
                decision = decision[3:-3].strip()

            # Enhanced JSON parsing with multiple fallback strategies
            decision_dict = self._try_parse_json_with_fallbacks(decision)

            # Extract and validate the parsed data
            explorer_name = decision_dict.get("explorer")
            task = decision_dict.get("task")

            if not explorer_name or not task:
                raise ValueError(f"Missing required fields. Got explorer='{explorer_name}', task='{task}'")

            # Verify if explorer is in available list
            if explorer_name not in available_explorers:
                raise ValueError(f"Invalid explorer: {explorer_name}. Available: {list(available_explorers.keys())}")

            return explorer_name, task

        except Exception as e:
            logging.info(f"[investigator] Unable to parse decision: {decision}")
            logging.info(f"[investigator] Error details: {e}")
            import traceback
            print(f"[investigator] Full traceback:")
            print(traceback.format_exc())
            raise

    def _try_parse_json_with_fallbacks(self, decision: str) -> dict:
        """Try multiple strategies to parse JSON decision"""
        import json
        import re

        decision_dict = None
        parsing_errors = []

        # Strategy 1: Direct JSON parsing
        try:
            decision_dict = json.loads(decision)
        except json.JSONDecodeError as e:
            parsing_errors.append(f"JSON parsing failed: {e}")

        # Strategy 2: Try to fix common JSON issues and parse again
        if decision_dict is None:
            try:
                fixed_decision = decision
                task_pattern = r'"task"\s*:\s*"([^"]*(?:"[^"]*)*)"'
                task_match = re.search(task_pattern, fixed_decision)
                if task_match:
                    task_content = task_match.group(1)
                    escaped_task = task_content.replace('"', '\\"')
                    fixed_decision = fixed_decision.replace(task_match.group(0), f'"task": "{escaped_task}"')
                decision_dict = json.loads(fixed_decision)
            except json.JSONDecodeError as e:
                parsing_errors.append(f"Fixed JSON parsing failed: {e}")

        # Strategy 3: Extract using regex patterns
        if decision_dict is None:
            try:
                explorer_match = re.search(r'"explorer"\s*:\s*"([^"]+)"', decision)
                task_match = re.search(r'"task"\s*:\s*"([^"]*(?:"[^"]*"[^"]*)*)"', decision)

                if explorer_match and task_match:
                    explorer_name = explorer_match.group(1)
                    task_content = task_match.group(1)
                    task_content = task_content.replace('\\"', '"')
                    decision_dict = {"explorer": explorer_name, "task": task_content}
                else:
                    parsing_errors.append("Regex extraction failed: could not find explorer or task")
            except Exception as e:
                parsing_errors.append(f"Regex extraction failed: {e}")

        # Strategy 4: Fall back to eval (risky but sometimes necessary)
        if decision_dict is None:
            try:
                decision_dict = eval(decision)
            except Exception as e:
                parsing_errors.append(f"Eval parsing failed: {e}")

        # If all parsing strategies failed, raise an exception
        if decision_dict is None:
            raise ValueError(f"All parsing strategies failed. Errors: {'; '.join(parsing_errors)}")

        return decision_dict

    async def _build_enriched_explorer_message(self, task: str, explorer_name: str) -> str:
        """Build enriched explorer message with tool recommendations"""
        explorer_msg = explorer_task_prompt.format(task=task)

        try:
            # Select manager by explorer
            if explorer_name == "metric_explorer":
                manager = get_metric_tool_ecosystem_manager()
                ds = "metric"
            elif explorer_name == "log_explorer":
                manager = get_log_tool_ecosystem_manager()
                ds = "log"
            else:
                manager = get_trace_tool_ecosystem_manager()
                ds = "trace"

            # get unified tool chain
            from agents import get_unified_tool_chain
            previous_tool_ids = get_unified_tool_chain()

            # use updated recommendation method, pass previous tool IDs
            recs = manager.recommend_tools_for_agent(
                task_description=task,
                previous_tool_ids=previous_tool_ids,
                retrieval_k=20,
                recommend_n=5,
                semantic_threshold=0.7
            )
            if ds:
                recs = [r for r in recs if r.get("data_source") == ds]

            # get allowed modules from config
            from config_manager import get_tool_files_with_env_override
            tool_files = get_tool_files_with_env_override()
            allowed_modules = {tool_files["metric_tools"], tool_files["log_tools"], tool_files["trace_tools"]}
            recs = [r for r in recs if r.get("module_path") in allowed_modules]

            if not recs:
                enriched_explorer_msg = explorer_msg + "No tool recommendations found, please generate new tool"
            else:
                lines = []
                for r in recs:
                    line = f"- {r.get('name')} | fn={r.get('module_path')}.{r.get('function_name')} | sim={r.get('semantic_similarity')} | quality={r.get('SBFL_quality_score')} | unified_score={r.get('SBFL_unified_score')} |hint={r.get('context_hint', '')}"
                    lines.append(line)
                suggestion_block = "\n\n[Tool Recommendations]\n[Sorting Rule: Arrange tools in descending order of their SBFL quality scores]\n" + "\n".join(lines)
                enriched_explorer_msg = explorer_msg + suggestion_block


        except Exception as e:
            print(f"Error building tool recommendations: {e}")
            import traceback
            print(traceback.format_exc())
            enriched_explorer_msg = explorer_msg

        # Add other explorer's execution result to message sent to explorer
        enriched_explorer_msg = self.explorer_notebook.enrich_message(
            enriched_explorer_msg,
            explorer_name
        )

        return enriched_explorer_msg

    def _should_retry_tool_call(self, response_text: str, explorer_name: str, has_tool_calls: bool, tool_call_success: bool) -> tuple:
        """
        Check if tool call should be retried
        Returns: (should_retry, retry_reason)
        """
        # Only check if there were actual tool calls
        if not has_tool_calls:
            return False, None

        # Check for execution errors using tool call success status
        if not tool_call_success:
            return True, "execution_error"

        # Check for anomaly event count issues
        try:
            is_valid, anomaly_count, error_message = check_anomaly_events_count(response_text, explorer_name)
            logging.info(f"[should_retry]: is_valid: {is_valid}, anomaly_count: {anomaly_count}, error_message: {error_message}")
            if not is_valid:
                logging.info(f"[{explorer_name}] Anomaly event count check failed: {error_message}")
                return True, "anomaly_count_invalid"
        except Exception as e:
            logging.info(f"[{explorer_name}] Error checking anomaly events: {e}")
            # If we can't check, don't retry for this reason
            pass

        return False, None

    def _check_need_tool_generation(self, response_text: str, explorer_name: str, has_tool_calls: bool, tool_call_success: bool) -> bool:
        """Check if we need tool generation due to errors or insufficient anomaly events"""
        if 'NEED_TOOL_GENERATION' in response_text:
            return True
        elif has_tool_calls and not tool_call_success:
            # Only check tool_call_success if there were actual tool calls
            return True
        elif has_tool_calls:
            # Also check if anomaly event count is still invalid after retries (only if there were tool calls)
            try:
                is_valid, anomaly_count, error_message = check_anomaly_events_count(response_text, explorer_name)
                logging.info(f"[should_retry2] is_valid: {is_valid}, anomaly_count: {anomaly_count}, error_message: {error_message}")
                if not is_valid:
                    logging.info(f"[{explorer_name}] Tool call still has anomaly event count issues after retries: {error_message}")
                    return True
            except Exception as e:
                logging.info(f"[{explorer_name}] Error checking anomaly events for tool generation decision: {e}")
        return False

    async def _execute_explorer_task(self, explorer_name: str, task: str, enriched_explorer_msg: str,
                                   scoring_points: str, generated_tools: list) -> tuple:
        """Execute explorer task with ReAct loop and error handling

        Returns:
            tuple: (result, task_used_generate_tool)
        """
        react_max_turns = 2
        react_turn = 0
        final_result_text = None
        next_message = enriched_explorer_msg[:10000]
        task_used_generate_tool = False

        while react_turn < react_max_turns:
            react_turn += 1
            response = await self.agents[explorer_name].on_messages(
                [TextMessage(content=next_message, source="investigator")],
                cancellation_token=CancellationToken(),
            )

            # Update statistics
            self.llm_call_count[explorer_name] += 1
            self.llm_call_count["total"] += 1

            prompt_tokens, completion_tokens = self._extract_token_usage(response, explorer_name, f"ReAct turn {react_turn}")
            self.token_usage[explorer_name]["prompt"] += prompt_tokens
            self.token_usage[explorer_name]["completion"] += completion_tokens
            self.token_usage[explorer_name]["total"] += prompt_tokens + completion_tokens
            self.token_usage["total"]["prompt"] += prompt_tokens
            self.token_usage["total"]["completion"] += completion_tokens
            self.token_usage["total"]["total"] += prompt_tokens + completion_tokens


            await self.print_llm_response(explorer_name, response, system_message=next_message)
            has_tool_calls, tool_call_success = self._record_tool_calls(explorer_name, response)

            response_text = response.chat_message.content or ""

            # Handle retries for tool call failures
            response_text = await self._handle_tool_call_retries(
                explorer_name, response, response_text, has_tool_calls, tool_call_success
            )

            # Check if we need tool generation
            need_tool_generation = self._check_need_tool_generation(
                response_text, explorer_name, has_tool_calls, tool_call_success
            )

            if need_tool_generation:
                next_message, task_used_generate_tool = await self._handle_tool_generation(
                    task, explorer_name, scoring_points, generated_tools
                )
                if next_message is None:  # Tool generation failed
                    break
                continue

            # Handle control tokens
            if 'TASK_COMPLETE' in response_text:
                final_result_text = response_text
                break

            if 'CONTINUE' in response_text:
                next_message = "CONTINUE"
                continue

            # If no explicit control token but anomaly_event is present, treat as completion
            if 'anomaly_event' in response_text:
                final_result_text = response_text
                break

            # Default: finish task
            break

        result = final_result_text or response.chat_message.content
        return result, task_used_generate_tool

    async def _handle_tool_call_retries(self, explorer_name: str, response, response_text: str,
                                      has_tool_calls: bool, tool_call_success: bool) -> str:
        """Handle retries for tool call failures"""
        retry_count = 0
        max_retries = 3

        should_retry_result, retry_reason = self._should_retry_tool_call(
            response_text, explorer_name, has_tool_calls, tool_call_success
        )

        logging.info(f'[should_retry_result]: {should_retry_result}, retry_reason: {retry_reason}')
        while should_retry_result and retry_count < max_retries:
            retry_count += 1

            if retry_reason == "execution_error":
                retry_message = "Tool call failed, please adjust parameters and retry"
                logging.info(f"[{explorer_name}] Call error occurred: {response_text}, retrying...")
            elif retry_reason == "anomaly_count_invalid":
                retry_message = "Tool call returned insufficient anomaly events, please adjust parameters to find more anomalies or try other tools"
                logging.info(f"[{explorer_name}] Anomaly event count issue: {response_text}")
            else:
                retry_message = "Tool call failed, please adjust parameters and retry"
                logging.info(f"[{explorer_name}] Unknown issue occurred: {response_text}")

            response = await self.agents[explorer_name].on_messages(
                [TextMessage(content=retry_message+"\n"+ "response(only first 1000 characters shown): "+response_text[:1000], source="investigator")],
                cancellation_token=CancellationToken(),
            )

            # Update statistics for retry
            self.llm_call_count[explorer_name] += 1
            self.llm_call_count["total"] += 1

            prompt_tokens, completion_tokens = self._extract_token_usage(response, explorer_name, "Retry")
            self.token_usage[explorer_name]["prompt"] += prompt_tokens
            self.token_usage[explorer_name]["completion"] += completion_tokens
            self.token_usage[explorer_name]["total"] += prompt_tokens + completion_tokens
            self.token_usage["total"]["prompt"] += prompt_tokens
            self.token_usage["total"]["completion"] += completion_tokens
            self.token_usage["total"]["total"] += prompt_tokens + completion_tokens

            await self.print_llm_response(explorer_name, response, system_message=f"Tool call failed ({retry_reason}), please adjust parameters and retry, or regenerate tool based on error information"+"\n"+response_text)

            has_tool_calls, tool_call_success = self._record_tool_calls(explorer_name, response)
            response_text = response.chat_message.content or ""

            # Check again for the next iteration
            should_retry_result, retry_reason = self._should_retry_tool_call(
                response_text, explorer_name, has_tool_calls, tool_call_success
            )

        return response_text

    async def _handle_tool_generation(self, task: str, explorer_name: str, scoring_points: str,
                                    generated_tools: list) -> tuple:
        """Handle tool generation when needed

        Returns:
            tuple: (next_message, task_used_generate_tool) or (None, True) if failed
        """
        tool_description = task
        try:
            tool_result, tool_info = await self.generate_tool(tool_description, explorer_name, scoring_points)
            generated_tool_execution_result = json.dumps(tool_result)

            # collect generated tools
            if tool_info and tool_info.get('blocks_to_save'):
                generated_tools.append(tool_info)
                logging.info(f"[Tool Collection] Collected tool info for {tool_info.get('coder_name')}")

            logging.info(f"[Task Completion Type] {explorer_name} used generate_tool for current task")

            # Provide execution results back to explorer to continue reasoning
            next_message = tool_execution_result_prompt.format(
                task=task,
                generated_tool_execution_result=generated_tool_execution_result
            )
            return next_message, True

        except Exception as _e:
            print(f"[{explorer_name}] Tool generation error: {str(_e)}")
            import traceback
            print(traceback.format_exc())
            return None, True

    def _handle_explorer_exception(self, e: Exception, explorer_name: str, response=None) -> str:
        """Handle exceptions that occur during explorer execution"""
        print(f"[{explorer_name}] Error occurred while processing response:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"Original response content:")
        try:
            if response:
                print(response.chat_message.content)
        except Exception:
            pass

        # If KeyError: '\ndata_source' error, try to parse response content
        if isinstance(e, KeyError) and "data_source" in str(e) and response:
            try:
                # Get original response content
                result = response.chat_message.content

                # If NEED_TOOL_GENERATION, use directly
                if "NEED_TOOL_GENERATION" in result:
                    pass
                else:
                    # Try to parse anomaly_event format response
                    import re
                    # Match anomaly_event = [...] format
                    match = re.search(r'anomaly_event\s*=\s*\[(.*?)\]', result, re.DOTALL)
                    if match:
                        # Extract matched content
                        content = match.group(1).strip()
                        # Format as valid JSON
                        content = content.replace("'", '"')
                        # Add parsing success marker
                        result = f"Successfully parsed response content: {content}"
                    else:
                        # If unable to parse, add error marker
                        result = f"Unable to parse response content, original content: {result}"
            except Exception as parse_error:
                print(f"Error occurred while trying to parse response content: {str(parse_error)}")
                result = f"Parse error: {str(parse_error)}, original response content: {response.chat_message.content if response else 'No response'}"
        else:
            try:
                result = response.chat_message.content if response else str(e)
            except Exception:
                result = str(e)

        return result

    async def run_investigation(self, investigator_msg: str, scoring_points: str = None) -> tuple:
        start_time = datetime.now()
        logging.info(f"[Time Statistics] Investigation phase start: {start_time}")

        investigation_results = []
        generated_tools = []  # collect generated tools
        max_rounds = 4  # Set maximum investigation rounds
        current_round = 0

        self._initialize_tool_library_count()

        # list all available explorers
        available_explorers = {
            "metric_explorer": "Used for querying and analyzing metric data.",
            "log_explorer": "Used for querying and analyzing log data.",
            "trace_explorer": "Used for querying and analyzing trace data."
        }
        
        while True:
            current_round += 1
            if current_round > max_rounds:
                logging.info(f"[investigator] Reached maximum investigation rounds {max_rounds}, forcing investigation to end")
                break

            # 1. Get investigator's decision
            try:
                explorer_name, task = await self._get_investigator_decision(investigator_msg, available_explorers)
                if explorer_name is None:  # Investigation complete
                    break
            except Exception as e:
                investigator_msg = exception_investigator_prompt_template.format(
                    available_explorers=list(available_explorers.keys())
                )
                continue

            # 2. Build enriched explorer message with tool recommendations
            enriched_explorer_msg = await self._build_enriched_explorer_message(task, explorer_name)

            # 3. Execute explorer task
            try:
                result, task_used_generate_tool = await self._execute_explorer_task(
                    explorer_name, task, enriched_explorer_msg, scoring_points, generated_tools
                )

                # Save result to explorer_notebook
                self.explorer_notebook.save_task(explorer_name, task)
                self.explorer_notebook.save_response(explorer_name, result)

                # record task completion type
                completion_type = 'g' if task_used_generate_tool else 'p'
                self.task_completion_types.append(completion_type)
                logging.info(f"[Task Completion Type] {explorer_name} task completed with type: {completion_type}")

                investigation_results.append({
                    "explorer": explorer_name,
                    "task": task,
                    "result": result
                })

            except Exception as e:
                result = self._handle_explorer_exception(e, explorer_name, None)

                # Save result to explorer_notebook
                self.explorer_notebook.save_task(explorer_name, task)
                self.explorer_notebook.save_response(explorer_name, result)

                # record task completion type (exception case)
                completion_type = 'g' if 'NEED_TOOL_GENERATION' in result else 'p'
                self.task_completion_types.append(completion_type)
                logging.info(f"[Task Completion Type] {explorer_name} task completed with type: {completion_type} (exception case)")

                investigation_results.append({
                    "explorer": explorer_name,
                    "task": task,
                    "result": result
                })

            investigator_msg = update_investigator_prompt_template.format(
                investigation_results=investigation_results
            )
        
        end_time = datetime.now()
        investigation_time = end_time - start_time
        self.timing["investigate"] = investigation_time
        logging.info(f"[Time Statistics] Investigation phase end: {end_time}, total time: {investigation_time}")
        logging.info(f"[Time Statistics] Coder part time: {self.timing['coder']}")
        logging.info(f"[Tool Collection] Collected {len(generated_tools)} tool(s) for potential persistence")

        return investigation_results, generated_tools

    async def run_diagnosis(self, user_query: str, queried_issue: Dict, reference_books: List[str], scoring_points: str = None) -> tuple:
        # Overall start time
        total_start_time = datetime.now()
        logging.info(f"[Time Statistics] Diagnosis process start: {total_start_time}")
        # Reset tool execution chain at the beginning of a diagnosis
        try:
            reset_tool_chains()
        except Exception as e:
            print(f"Warning: failed to reset tool chain: {e}")
        
        # 1. Planning Stage
        plan_start_time = datetime.now()
        logging.info(f"[Time Statistics] Planning phase start: {plan_start_time}")
        
        planning_msg = planning_msg_template.format(
            planner_role_description=planner_role_description,
            user_query=user_query,
            background=background
        )
        
        response = await self.agents["planner"].on_messages(
            [TextMessage(content=planning_msg, source="user")],
            cancellation_token=CancellationToken(),
        )
        await self.print_llm_response("planner", response, system_message=planning_msg)
        diagnosis_plan = response.chat_message.content
        # Enhanced token statistics with debugging for planner
        prompt_tokens, completion_tokens = self._extract_token_usage(response, "planner", "main call")

        self.token_usage["planner"]["prompt"] += prompt_tokens
        self.token_usage["planner"]["completion"] += completion_tokens
        self.token_usage["planner"]["total"] += prompt_tokens + completion_tokens

        self.token_usage["total"]["prompt"] += prompt_tokens
        self.token_usage["total"]["completion"] += completion_tokens
        self.token_usage["total"]["total"] += prompt_tokens + completion_tokens

        
        
        plan_end_time = datetime.now()
        plan_time = plan_end_time - plan_start_time
        self.timing["plan"] = plan_time
        
        investigation_msg = investigation_msg_template.format(
            user_query=user_query,
            diagnosis_plan=diagnosis_plan
        )
        
        diagnosis_events, generated_tools = await self.run_investigation(investigation_msg, scoring_points)
        
        # 3. Reasoning Stage
        reason_start_time = datetime.now()
        
        reasoning_msg = reasoning_msg_template.format(
            user_query=user_query,
            queried_issue=queried_issue,
            diagnosis_plan=diagnosis_plan,
            diagnosis_events=diagnosis_events
        )
        
        response = await self.agents["reasoner"].on_messages(
            [TextMessage(content=reasoning_msg, source="investigator")],
            cancellation_token=CancellationToken(),
        )
        # Increase LLM call count and token statistics - reasoner
        self.llm_call_count["reasoner"] += 1
        self.llm_call_count["total"] += 1
        
        # Enhanced token statistics with debugging for reasoner
        prompt_tokens, completion_tokens = self._extract_token_usage(response, "reasoner", "main call")

        self.token_usage["reasoner"]["prompt"] += prompt_tokens
        self.token_usage["reasoner"]["completion"] += completion_tokens
        self.token_usage["reasoner"]["total"] += prompt_tokens + completion_tokens

        self.token_usage["total"]["prompt"] += prompt_tokens
        self.token_usage["total"]["completion"] += completion_tokens
        self.token_usage["total"]["total"] += prompt_tokens + completion_tokens

        
        await self.print_llm_response("reasoner", response, system_message=reasoning_msg)
        root_cause = response.chat_message.content
        
        reason_end_time = datetime.now()
        reason_time = reason_end_time - reason_start_time
        self.timing["reason"] = reason_time
        
        # Overall end time
        total_end_time = datetime.now()
        total_time = total_end_time - total_start_time
        self.timing["total"] = total_time
        logging.info(f"[Time Statistics] Diagnosis process end: {total_end_time}, total time: {total_time}")
        logging.info(f"[Time Statistics] Time usage by phase:")
        logging.info(f"  - Planning phase: {self.timing['plan']}")
        logging.info(f"  - Investigation phase: {self.timing['investigate']}")
        logging.info(f"    - Coder part: {self.timing['coder']}")
        logging.info(f"  - Reasoning phase: {self.timing['reason']}")
        
        logging.info(f"[LLM Call Statistics] Calls by phase:")
        for agent, count in self.llm_call_count.items():
            if count > 0:
                logging.info(f"  - {agent}: {count}")
        logging.info(f"  - Total: {self.llm_call_count['total']}")
        
        # Output token statistics
        logging.info(f"[Token Statistics] Token usage by component:")
        for agent, usage in self.token_usage.items():
            if usage["total"] > 0 and agent != "total":
                logging.info(f"  - {agent}: input={usage['prompt']}, output={usage['completion']}, total={usage['total']}")
        logging.info(f"  - Total: input={self.token_usage['total']['prompt']}, output={self.token_usage['total']['completion']}, total={self.token_usage['total']['total']}")
        
        return {
            "diagnosis_plan": diagnosis_plan,
            "diagnosis_events": diagnosis_events,
            "root_cause": root_cause,
            "task_completion_types": self.task_completion_types,
            "tool_library_count": self.tool_library_count
        }, generated_tools

    def _record_tool_calls(self, explorer_name, response):
        """Parse response.inner_messages and record tool function calls once per response.

        Returns:
            tuple: (has_tool_calls, overall_success)
                - has_tool_calls: True if any tool calls were made, False otherwise
                - overall_success: True if all tool calls were successful, False if any failed
        """
        overall_success = True
        has_tool_calls = False
        try:
            if hasattr(response, 'inner_messages') and response.inner_messages:
                # collect FunctionCall and FunctionExecutionResult
                function_calls = {}  # {function_name: call_info}
                execution_results = {}  # {function_name: (is_success, call_id)}

                for m in response.inner_messages:
                    message_type = getattr(m, 'type', str(type(m).__name__))

                    # handle ToolCallRequestEvent (contains FunctionCall)
                    if 'ToolCallRequestEvent' in message_type or 'RequestEvent' in message_type:
                        try:
                            items = getattr(m, 'content', None)
                            if isinstance(items, list):
                                for it in items:
                                    if hasattr(it, '__class__') and 'FunctionCall' in str(it.__class__):
                                        name = getattr(it, 'name', None)
                                        call_id = getattr(it, 'id', None)
                                        if name:
                                            function_calls[name] = {'call_id': call_id, 'type': 'request'}
                        except Exception as e:
                            logging.info(f"[Debug] Error processing ToolCallRequestEvent: {e}")

                    # handle ToolCallExecutionEvent (contains FunctionExecutionResult)
                    elif 'ToolCallExecutionEvent' in message_type or 'ExecutionEvent' in message_type:
                        try:
                            items = getattr(m, 'content', None)
                            if isinstance(items, list):
                                for it in items:
                                    if hasattr(it, '__class__') and 'FunctionExecutionResult' in str(it.__class__):
                                        name = getattr(it, 'name', None)
                                        call_id = getattr(it, 'call_id', None)
                                        is_error = getattr(it, 'is_error', False)
                                        if name:
                                            is_success = not is_error
                                            execution_results[name] = (is_success, call_id)
                                            logging.info(f"[Debug] Found FunctionExecutionResult: {name} (call_id: {call_id}, is_error: {is_error}, is_success: {is_success})")
                        except Exception as e:
                            logging.info(f"[Debug] Error processing ToolCallExecutionEvent: {e}")

                    # handle FunctionExecutionResultMessage (directly contains FunctionExecutionResult)
                    elif 'FunctionExecutionResultMessage' in message_type:
                        try:
                            items = getattr(m, 'content', None)
                            if isinstance(items, list):
                                for it in items:
                                    name = getattr(it, 'name', None)
                                    call_id = getattr(it, 'call_id', None)
                                    is_error = getattr(it, 'is_error', False)
                                    if name:
                                        is_success = not is_error
                                        execution_results[name] = (is_success, call_id)
                                        logging.info(f"[Debug] Found FunctionExecutionResult in Message: {name} (call_id: {call_id}, is_error: {is_error}, is_success: {is_success})")
                        except Exception as e:
                            logging.info(f"[Debug] Error processing FunctionExecutionResultMessage: {e}")

                # record tool calls: prefer execution results, if none, use request calls
                recorded_functions = set()

                # first record tool calls with execution results
                for func_name, (is_success, call_id) in execution_results.items():
                    if func_name not in recorded_functions:
                        from agents import record_tool_call_with_result
                        logging.info(f"[Debug] Recording tool call with execution result: {explorer_name}.{func_name}, success: {is_success}")
                        record_tool_call_with_result(explorer_name, func_name, is_success)
                        recorded_functions.add(func_name)
                        has_tool_calls = True  # Mark that we have tool calls
                        # Update overall success status
                        if not is_success:
                            overall_success = False

                # then record tool calls without execution results (default success)
                for func_name in function_calls.keys():
                    if func_name not in recorded_functions:
                        from agents import record_tool_call_with_result
                        logging.info(f"[Debug] Recording tool call without execution result: {explorer_name}.{func_name}, success: True (default)")
                        record_tool_call_with_result(explorer_name, func_name, True)
                        recorded_functions.add(func_name)
                        has_tool_calls = True  # Mark that we have tool calls

        except Exception as _e:
            logging.info(f"Warning: failed to record tool call: {_e}")
            import traceback
            print(traceback.format_exc())
            overall_success = False

        return has_tool_calls, overall_success

    def _extract_token_usage(self, response, agent_name: str, context: str = "") -> tuple:
        """
        Enhanced token usage extraction with debugging information
        Returns: (prompt_tokens, completion_tokens)
        """
        try:
            # Method 1: Try standard chat_message.models_usage
            if hasattr(response, 'chat_message') and response.chat_message:
                if hasattr(response.chat_message, 'models_usage') and response.chat_message.models_usage:
                    prompt_tokens = getattr(response.chat_message.models_usage, 'prompt_tokens', 0)
                    completion_tokens = getattr(response.chat_message.models_usage, 'completion_tokens', 0)

                    if prompt_tokens > 0 or completion_tokens > 0:
                        return prompt_tokens, completion_tokens


                # Method 2: Try to find usage in chat_message itself
                if hasattr(response.chat_message, 'usage'):
                    prompt_tokens = getattr(response.chat_message.usage, 'prompt_tokens', 0)
                    completion_tokens = getattr(response.chat_message.usage, 'completion_tokens', 0)
                    return prompt_tokens, completion_tokens

            # Method 3: Try to find usage in response itself
            if hasattr(response, 'usage'):
                prompt_tokens = getattr(response.usage, 'prompt_tokens', 0)
                completion_tokens = getattr(response.usage, 'completion_tokens', 0)
                return prompt_tokens, completion_tokens

            # Method 4: Check inner_messages for usage information
            if hasattr(response, 'inner_messages') and response.inner_messages:
                for msg in response.inner_messages:
                    if hasattr(msg, 'models_usage') and msg.models_usage:
                        prompt_tokens = getattr(msg.models_usage, 'prompt_tokens', 0)
                        completion_tokens = getattr(msg.models_usage, 'completion_tokens', 0)
                        if prompt_tokens > 0 or completion_tokens > 0:
                            return prompt_tokens, completion_tokens

            return 0, 0

        except Exception as e:
            logging.info(f"[Token Debug] {agent_name} {context}: Error extracting token usage: {e}")
            import traceback
            print(traceback.format_exc())
            return 0, 0

    async def print_llm_response(self, agent_name, response, system_message=None):
        print(f"--------------------------------{agent_name}--------------------------------")
        print(f"[{agent_name}] response:")
        print(response.chat_message.content)
        print("-----------------------------------------------------------------------------")



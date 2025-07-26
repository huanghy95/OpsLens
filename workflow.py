from typing import Dict, List, Optional
from agents import *
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken, SingleThreadedAgentRuntime
import pprint
from autogen_core import AgentId
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from coder import *
import os
import json
from datetime import datetime, timedelta
from code_utils import save_code_blocks, load_code_blocks, save_code_as_functions

from coder import MetricCoder, LogCoder, TraceCoder, Coder

from prompt import get_prompt_module
from prompt.WorkflowPrompt import *

prompt_module = get_prompt_module()

background = getattr(prompt_module, "background", "")
data_description = getattr(prompt_module, "data_description", {})
diagnosis_plan = getattr(prompt_module, "diagnosis_plan", "")






import re


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

    @classmethod
    async def create(cls):
        """Asynchronous factory method to create and initialize DiagnosisWorkflow instance"""
        workflow = cls()
        
        # Initialize memory
        from agents import initialize_memory
        await initialize_memory()
        
        # 1. Create and start executor first
        workflow.docker_executor = DockerCommandLineCodeExecutor(work_dir="coding",auto_remove=False,container_name='opslens')
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
        print(f"\n{'='*50}")
        print(f"[LLM Call Statistics Summary]")
        for agent, count in self.llm_call_count.items():
            if count > 0 and agent != "total":
                print(f"  - {agent}: {count}")
        print(f"  - Total: {self.llm_call_count['total']}")
        
        # Output Token usage statistics
        print(f"\n[Token Usage Statistics Summary]")
        for agent, usage in self.token_usage.items():
            if usage["total"] > 0 and agent != "total":
                print(f"  - {agent}: input={usage['prompt']}, output={usage['completion']}, total={usage['total']}")
        print(f"  - Total: input={self.token_usage['total']['prompt']}, output={self.token_usage['total']['completion']}, total={self.token_usage['total']['total']}")

        
        # Output time usage statistics
        print(f"[Time Statistics] Diagnosis process end, total time: {self.timing['total']}")
        print(f"[Time Statistics] Time usage by phase:")
        print(f"  - Planning phase: {self.timing['plan']}")
        print(f"  - Investigation phase: {self.timing['investigate']}")
        print(f"    - Coder part: {self.timing['coder']}")
        print(f"  - Reasoning phase: {self.timing['reason']}")
        
        print(f"{'='*50}\n")

    async def __aenter__(self):
        """Async context manager entry point"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit point"""
        await self.cleanup()

    async def generate_tool(self, task_description: str, explorer_name: str) -> str:
        start_time = datetime.now()
        try:
            self.runtime.start()
        except Exception as e:
            print(f"Runtime start warning: {e}")
        
        coder_name = explorer_name[:-8] + "coder"
        
       
        # Add coder's identifier when sending messages
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
                    print(f"\nCode block {i}:")
                    print(f"Language: {code_block.language}")
                    print("Code content:")
                    print(code_block.code)
                    blocks_to_save.append(code_block)
                else:
                    print(f"Skipping non-Python code block, language: {code_block.language}")
            
            if blocks_to_save:
                save_code_blocks(blocks_to_save)
                save_code_as_functions(blocks_to_save, task_description)
            
            execution_result = Executor.get_execution_result()
            print("======generate_tool execution_result=======")
            pprint.pprint(execution_result)
            print("======generate_tool execution_result=======")
            self.coder_notebook.save_response(coder_name, execution_result)
            
            # Update coder time
            end_time = datetime.now()
            self.timing["coder"] += end_time - start_time
            print(f"[Time Statistics] {coder_name} tool generation time: {end_time - start_time}")
            
            return execution_result
        
        except Exception as e:
            # Update coder time (even if error occurs)
            end_time = datetime.now()
            self.timing["coder"] += end_time - start_time
            print(f"[Time Statistics] {coder_name} tool generation time (error): {end_time - start_time}")
            
            print(f"Error in generate_tool for {coder_name}: {str(e)}")
            print(f"Error details: {type(e).__name__}")  # Add more detailed error information
            return None


    async def run_investigation(self, investigator_msg: str) -> List[Dict]:
        start_time = datetime.now()
        print(f"[Time Statistics] Investigation phase start: {start_time}")
        
        investigation_results = []
        max_rounds = 5  # Set maximum investigation rounds
        current_round = 0
        
        # List all available explorers
        available_explorers = {
            "metric_explorer": "Used for querying and analyzing metric data.",
            "log_explorer": "Used for querying and analyzing log data.",
            "trace_explorer": "Used for querying and analyzing trace data."
        }
        
        while True:
            current_round += 1
            if current_round > max_rounds:
                print(f"[investigator] Reached maximum investigation rounds {max_rounds}, forcing investigation to end")
                break
            
            # 1. Investigator decides next investigation direction
            investigator_prompt = investigator_prompt_template.format(
                investigator_msg=investigator_msg,
                available_explorers=available_explorers
            )
            
            response = await self.agents["investigator"].on_messages(
                [TextMessage(content=investigator_prompt, source="user")],
                cancellation_token=CancellationToken(),
            )
            # Increase LLM call count and token statistics - investigator
            self.llm_call_count["investigator"] += 1
            self.llm_call_count["total"] += 1
            
            # Check if token statistics exist
            if hasattr(response, 'chat_message') and response.chat_message:
                if hasattr(response.chat_message, 'models_usage'):
                    prompt_tokens = getattr(response.chat_message.models_usage, 'prompt_tokens', 0)
                    completion_tokens = getattr(response.chat_message.models_usage, 'completion_tokens', 0)
                    
                    self.token_usage["investigator"]["prompt"] += prompt_tokens
                    self.token_usage["investigator"]["completion"] += completion_tokens
                    self.token_usage["investigator"]["total"] += prompt_tokens + completion_tokens
                    
                    self.token_usage["total"]["prompt"] += prompt_tokens
                    self.token_usage["total"]["completion"] += completion_tokens
                    self.token_usage["total"]["total"] += prompt_tokens + completion_tokens
                    
                    print(f"[Token Statistics] investigator: prompt={prompt_tokens}, completion={completion_tokens}, current total={self.token_usage['investigator']['total']}")
            
            await self.print_llm_response("investigator", response)
            decision = response.chat_message.content

                
            # 2. Parse investigator's decision
            try:
                if "INVESTIGATION_COMPLETE" in decision:
                    break
                    
                # Try to remove possible leading/trailing whitespace characters
                decision = decision.strip()
                
                # Handle markdown code block format (```json ... ```)
                if decision.startswith("```json") and decision.endswith("```"):
                    # Extract JSON content from markdown code block
                    decision = decision[7:-3].strip()  # Remove ```json and ```
                elif decision.startswith("```") and decision.endswith("```"):
                    # Handle generic code block format (``` ... ```)
                    decision = decision[3:-3].strip()  # Remove ``` and ```
                
                # Try JSON parsing first, then fall back to eval
                try:
                    import json
                    decision_dict = json.loads(decision)
                except json.JSONDecodeError:
                    # Fall back to eval for non-JSON format
                    decision_dict = eval(decision)
                
                explorer_name = decision_dict["explorer"]
                task = decision_dict["task"]
                # Verify if explorer is in available list
                if explorer_name not in available_explorers:
                    raise ValueError(f"Invalid explorer: {explorer_name}")  

            except Exception as e:
                print(f"[investigator] Unable to parse decision: {decision}")
                pprint.pprint(e)
               
                investigator_msg = exception_investigator_prompt_template.format(
                    available_explorers=list(available_explorers.keys())
                )
                # TODO: Previous error information
                continue
                
            # 3. Call selected explorer to perform investigation
            explorer_msg = explorer_task_prompt.format(task=task)
            
            # Add other explorer's execution result to message sent to explorer
            enriched_explorer_msg = self.explorer_notebook.enrich_message(
                explorer_msg, 
                explorer_name
            )
            
            response = await self.agents[explorer_name].on_messages(
                [TextMessage(content=enriched_explorer_msg, source="investigator")],
                cancellation_token=CancellationToken(),
            )
            # Increase LLM call count and token statistics - explorer
            self.llm_call_count[explorer_name] += 1
            self.llm_call_count["total"] += 1
            
            # Check if token statistics exist
            if hasattr(response, 'chat_message') and response.chat_message:
                if hasattr(response.chat_message, 'models_usage'):
                    prompt_tokens = getattr(response.chat_message.models_usage, 'prompt_tokens', 0)
                    completion_tokens = getattr(response.chat_message.models_usage, 'completion_tokens', 0)
                    
                    self.token_usage[explorer_name]["prompt"] += prompt_tokens
                    self.token_usage[explorer_name]["completion"] += completion_tokens
                    self.token_usage[explorer_name]["total"] += prompt_tokens + completion_tokens
                    
                    self.token_usage["total"]["prompt"] += prompt_tokens
                    self.token_usage["total"]["completion"] += completion_tokens
                    self.token_usage["total"]["total"] += prompt_tokens + completion_tokens
                    
                    print(f"[Token Statistics] {explorer_name}: prompt={prompt_tokens}, completion={completion_tokens}, current total={self.token_usage[explorer_name]['total']}")
            
            await self.print_llm_response(explorer_name, response)
            
            # Check if need to generate new tool
            try:
                # Check if response contains need_tool_generate
                response_text = response.chat_message.content
                need_tool = False
                tool_description = None
                retry_count = 0
                max_retries = 3
                while 'Error' in response_text and retry_count < max_retries:
                    retry_count += 1
                    print(f"[{explorer_name}] Call error occurred: {response_text}")
                    response = await self.agents[explorer_name].on_messages(
                        [TextMessage(content="Tool call failed, please regenerate tool based on error information"+"\n"+response_text, source="investigator")],
                        cancellation_token=CancellationToken(),
                    )
                    # Increase LLM call count and token statistics - explorer (error retry)
                    self.llm_call_count[explorer_name] += 1
                    self.llm_call_count["total"] += 1
                    
                    # Check if token statistics exist
                    if hasattr(response, 'chat_message') and response.chat_message:
                        if hasattr(response.chat_message, 'models_usage'):
                            prompt_tokens = getattr(response.chat_message.models_usage, 'prompt_tokens', 0)
                            completion_tokens = getattr(response.chat_message.models_usage, 'completion_tokens', 0)
                            
                            self.token_usage[explorer_name]["prompt"] += prompt_tokens
                            self.token_usage[explorer_name]["completion"] += completion_tokens
                            self.token_usage[explorer_name]["total"] += prompt_tokens + completion_tokens
                            
                            self.token_usage["total"]["prompt"] += prompt_tokens
                            self.token_usage["total"]["completion"] += completion_tokens
                            self.token_usage["total"]["total"] += prompt_tokens + completion_tokens
                            
                            print(f"[Token Statistics] {explorer_name}(Retry): prompt={prompt_tokens}, completion={completion_tokens}, current total={self.token_usage[explorer_name]['total']}")
                    
                    await self.print_llm_response(explorer_name, response)
                    response_text = response.chat_message.content
                # Try to extract need_tool_generate and tool_description from response
                if 'NEED_TOOL_GENERATION' in response_text:
                
                    need_tool = True
                    tool_description = task
                
                if need_tool and tool_description:
                    # Generate new tool
                    generated_tool_execution_result = json.dumps(await self.generate_tool(tool_description, explorer_name))
                    
                    # Re-execute investigation task
                    enriched_explorer_msg = tool_execution_result_prompt.format(
                        task=task, 
                        generated_tool_execution_result=generated_tool_execution_result
                    )
                    
                    response = await self.agents[explorer_name].on_messages(
                        [TextMessage(content=enriched_explorer_msg, source="investigator")],
                        cancellation_token=CancellationToken(),
                    )
                    self.llm_call_count[explorer_name] += 1
                    self.llm_call_count["total"] += 1
                    
                    # Check if token statistics exist
                    if hasattr(response, 'chat_message') and response.chat_message:
                        if hasattr(response.chat_message, 'models_usage'):
                            prompt_tokens = getattr(response.chat_message.models_usage, 'prompt_tokens', 0)
                            completion_tokens = getattr(response.chat_message.models_usage, 'completion_tokens', 0)
                            
                            self.token_usage[explorer_name]["prompt"] += prompt_tokens
                            self.token_usage[explorer_name]["completion"] += completion_tokens
                            self.token_usage[explorer_name]["total"] += prompt_tokens + completion_tokens
                            
                            self.token_usage["total"]["prompt"] += prompt_tokens
                            self.token_usage["total"]["completion"] += completion_tokens
                            self.token_usage["total"]["total"] += prompt_tokens + completion_tokens
                            
                            print(f"[Token Statistics] {explorer_name}(Tool generated): prompt={prompt_tokens}, completion={completion_tokens}, current total={self.token_usage[explorer_name]['total']}")
                    
                    await self.print_llm_response(explorer_name, response)
                
                result = response.chat_message.content
                # Save result to explorer_notebook
                self.explorer_notebook.save_task(explorer_name, task)
                self.explorer_notebook.save_response(explorer_name, result)
                
                investigation_results.append({
                    "explorer": explorer_name,
                    "task": task,
                    "result": result
                })
                
            except Exception as e:
                print(f"[{explorer_name}] Error occurred while processing response:")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                print(f"Original response content:")
                print(response.chat_message.content)
                
                # If KeyError: '\ndata_source' error, try to parse response content
                if isinstance(e, KeyError) and "data_source" in str(e):
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
                        result = f"Parse error: {str(parse_error)}, original response content: {response.chat_message.content}"
                else:
                    result = response.chat_message.content
                
                # Save result to explorer_notebook
                self.explorer_notebook.save_task(explorer_name, task)
                self.explorer_notebook.save_response(explorer_name, result)
                
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
        print(f"[Time Statistics] Investigation phase end: {end_time}, total time: {investigation_time}")
        print(f"[Time Statistics] Coder part time: {self.timing['coder']}")
            
        return investigation_results

    async def run_diagnosis(self, user_query: str, queried_issue: Dict, reference_books: List[str]):
        # Overall start time
        total_start_time = datetime.now()
        print(f"[Time Statistics] Diagnosis process start: {total_start_time}")
        
        # 1. Planning Stage
        plan_start_time = datetime.now()
        print(f"[Time Statistics] Planning phase start: {plan_start_time}")
        
        planning_msg = planning_msg_template.format(
            planner_role_description=planner_role_description,
            user_query=user_query,
            background=background
        )
        
        response = await self.agents["planner"].on_messages(
            [TextMessage(content=planning_msg, source="user")],
            cancellation_token=CancellationToken(),
        )
        await self.print_llm_response("planner", response)
        diagnosis_plan = response.chat_message.content
        if hasattr(response, 'chat_message') and response.chat_message:
            if hasattr(response.chat_message, 'models_usage'):
                prompt_tokens = getattr(response.chat_message.models_usage, 'prompt_tokens', 0)
                completion_tokens = getattr(response.chat_message.models_usage, 'completion_tokens', 0)
                
                self.token_usage["planner"]["prompt"] += prompt_tokens
                self.token_usage["planner"]["completion"] += completion_tokens
                self.token_usage["planner"]["total"] += prompt_tokens + completion_tokens
                
                self.token_usage["total"]["prompt"] += prompt_tokens
                self.token_usage["total"]["completion"] += completion_tokens
                self.token_usage["total"]["total"] += prompt_tokens + completion_tokens
                
                print(f"[Token Statistics] planner: prompt={prompt_tokens}, completion={completion_tokens}, current total={self.token_usage['investigator']['total']}")     
        
        
        plan_end_time = datetime.now()
        plan_time = plan_end_time - plan_start_time
        self.timing["plan"] = plan_time
        print(f"[Time Statistics] Planning phase end: {plan_end_time}, time used: {plan_time}")
        
        investigation_msg = investigation_msg_template.format(
            user_query=user_query,
            diagnosis_plan=diagnosis_plan
        )
        
        diagnosis_events = await self.run_investigation(investigation_msg)
        
        # 3. Reasoning Stage
        reason_start_time = datetime.now()
        print(f"[Time Statistics] Reasoning phase start: {reason_start_time}")
        
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
        
        # Check if token statistics exist
        if hasattr(response, 'chat_message') and response.chat_message:
            if hasattr(response.chat_message, 'models_usage'):
                prompt_tokens = getattr(response.chat_message.models_usage, 'prompt_tokens', 0)
                completion_tokens = getattr(response.chat_message.models_usage, 'completion_tokens', 0)
                
                self.token_usage["reasoner"]["prompt"] += prompt_tokens
                self.token_usage["reasoner"]["completion"] += completion_tokens
                self.token_usage["reasoner"]["total"] += prompt_tokens + completion_tokens
                
                self.token_usage["total"]["prompt"] += prompt_tokens
                self.token_usage["total"]["completion"] += completion_tokens
                self.token_usage["total"]["total"] += prompt_tokens + completion_tokens
                
                print(f"[Token Statistics] reasoner: prompt={prompt_tokens}, completion={completion_tokens}, current total={self.token_usage['reasoner']['total']}")
        
        await self.print_llm_response("reasoner", response)
        root_cause = response.chat_message.content
        
        reason_end_time = datetime.now()
        reason_time = reason_end_time - reason_start_time
        self.timing["reason"] = reason_time
        print(f"[Time Statistics] Reasoning phase end: {reason_end_time}, time used: {reason_time}")
        
        # Overall end time
        total_end_time = datetime.now()
        total_time = total_end_time - total_start_time
        self.timing["total"] = total_time
        print(f"[Time Statistics] Diagnosis process end: {total_end_time}, total time: {total_time}")
        print(f"[Time Statistics] Time usage by phase:")
        print(f"  - Planning phase: {self.timing['plan']}")
        print(f"  - Investigation phase: {self.timing['investigate']}")
        print(f"    - Coder part: {self.timing['coder']}")
        print(f"  - Reasoning phase: {self.timing['reason']}")
        
        print(f"[LLM Call Statistics] Calls by phase:")
        for agent, count in self.llm_call_count.items():
            if count > 0:
                print(f"  - {agent}: {count}")
        print(f"  - Total: {self.llm_call_count['total']}")
        
        # Output token statistics
        print(f"[Token Statistics] Token usage by component:")
        for agent, usage in self.token_usage.items():
            if usage["total"] > 0 and agent != "total":
                print(f"  - {agent}: input={usage['prompt']}, output={usage['completion']}, total={usage['total']}")
        print(f"  - Total: input={self.token_usage['total']['prompt']}, output={self.token_usage['total']['completion']}, total={self.token_usage['total']['total']}")
        
        return {
            "diagnosis_plan": diagnosis_plan,
            "diagnosis_events": diagnosis_events,
            "root_cause": root_cause
        } 
    
    
    async def print_llm_response(self, agent_name, response):
        print(f"--------------------------------{agent_name}--------------------------------")
        print(f"[{agent_name}] response:")
        # pprint.pprint(response.chat_message.content)
        print(response.chat_message.content)
        print("-----------------------------------------------------------------------------")
    
 
import re
import logging
from datetime import datetime   
from dataclasses import dataclass
from typing import List
from autogen_core import AgentId
from autogen_core import MessageContext, RoutedAgent, default_subscription, message_handler
from autogen_core.code_executor import CodeBlock, CodeExecutor
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    LLMMessage,
    SystemMessage,
    UserMessage,
)


from prompt import get_prompt_module
from config_manager import get_prompt_type_with_env_override
from NoteBook import NotebookSystem
from coder_memory import retrieve_experience_memories, save_refinement_experience_to_memory

prompt_type = get_prompt_type_with_env_override()
prompt_module = get_prompt_module(prompt_type)

metric_system_coder = getattr(prompt_module, "metric_system_coder", "")
log_system_coder = getattr(prompt_module, "log_system_coder", "")
trace_system_coder = getattr(prompt_module, "trace_system_coder", "")
log_refine_rules = getattr(prompt_module, "log_refine_rules", "")
metric_refine_rules = getattr(prompt_module, "metric_refine_rules", "")
trace_refine_rules = getattr(prompt_module, "trace_refine_rules", "")
metric_anomaly_events_max_count = getattr(prompt_module, "metric_anomaly_events_max_count", 30)
metric_anomaly_events_min_count = getattr(prompt_module, "metric_anomaly_events_min_count", 5)
trace_anomaly_events_max_count = getattr(prompt_module, "trace_anomaly_events_max_count", 5)
trace_anomaly_events_min_count = getattr(prompt_module, "trace_anomaly_events_min_count", 0)
log_anomaly_events_max_count = getattr(prompt_module, "log_anomaly_events_max_count", 3)
log_anomaly_events_min_count = getattr(prompt_module, "log_anomaly_events_min_count", 0)




@dataclass
class Message:
    content: str


@default_subscription
class Coder(RoutedAgent):
    _instances = {}  
    _notebook = NotebookSystem()  
    _llm_call_count = 0
    _token_usage = {
        "prompt": 0,
        "completion": 0,
        "total": 0
    }
    
    def __init__(self, model_client: ChatCompletionClient, name: str = "coder") -> None:
        super().__init__("An Coder agent.")
        self._model_client = model_client
        self._name = name  
        self._chat_history: List[LLMMessage] = [
            SystemMessage(
                content=metric_system_coder,
            )
        ]
        Coder._instances[name if name else id(self)] = self  

    @staticmethod
    def get_chat_history(instance_name=None):
        if instance_name and instance_name in Coder._instances:  
            return Coder._instances[instance_name]._chat_history
        elif len(Coder._instances) > 0:
            return next(iter(Coder._instances.values()))._chat_history
        return []
    
    @staticmethod
    def get_llm_call_count():
        """Get LLM call count"""
        return Coder._llm_call_count
    
    @staticmethod
    def get_token_usage():
        """Get token usage statistics"""
        return Coder._token_usage


           

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        enriched_content = Coder._notebook.enrich_message(
            message.content, 
            self._name
        )
        
        import re
        is_success = False
        success_match = re.search(r'<success>(.*?)</success>', enriched_content, re.DOTALL)
        if success_match:
            is_success = True
            
        task_match = re.search(r'<task>(.*?)</task>', enriched_content, re.DOTALL)
        if task_match:
            task_content = task_match.group(1).strip()
            Coder._notebook.save_task(self._name, task_content)
            
            try:
                experience_memories = await retrieve_experience_memories(self._name, task_content, limit=3)
                if experience_memories:
                    memory_context = "\n\n=== Historical Refinement Experiences ===\n"
                    memory_context += "Based on similar tasks, here are some previous experiences to learn from:\n\n"
                    for i, memory in enumerate(experience_memories, 1):
                        memory_context += f"Experience {i}:\n{memory}\n\n"
                    memory_context += "=== End of Historical Experiences ===\n\n"
                    
                    enriched_content = enriched_content + memory_context
                    logging.info(f"[CoderMemory] Retrieved {len(experience_memories)} historical refinement experiences for {self._name}")
                else:
                    logging.info(f"[CoderMemory] No historical refinement experiences found for {self._name}")
            except Exception as e:
                logging.debug(f"[CoderMemory] Failed to retrieve experiences for {self._name}: {e}")
        
        self._chat_history.append(UserMessage(content=enriched_content, source="user"))
        
        if not is_success:
            # Increase LLM call count
            Coder._llm_call_count += 1
            logging.info(f"[LLM Call Statistics] {self._name} called LLM, Total: {Coder._llm_call_count}")
            
            result = await self._model_client.create(self._chat_history)
            
            # Check and update token usage statistics
            if hasattr(result, 'usage'):
                # RequestUsage format token statistics
                prompt_tokens = getattr(result.usage, 'prompt_tokens', 0)
                completion_tokens = getattr(result.usage, 'completion_tokens', 0)
                
                # Calculate total tokens
                total_tokens = prompt_tokens + completion_tokens
                
                Coder._token_usage["prompt"] += prompt_tokens
                Coder._token_usage["completion"] += completion_tokens
                Coder._token_usage["total"] += total_tokens
                
                logging.info(f"[Token Statistics] {self._name}: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}, cumulative={Coder._token_usage['total']}")
            print(f"\n{'-'*80}\n{self._name} Assistant:\n{result.content}")
            self._chat_history.append(AssistantMessage(content=result.content, source="assistant"))
        
            compressed_output = compress_duplicate_messages(result.content)
            Coder._notebook.save_response(self._name, compressed_output)
            
            coder_message = Message(self._name + ":\n" + result.content)
        

            await self.send_message(
                coder_message,
                recipient=AgentId("executor", "default")
            )



def extract_markdown_code_blocks(markdown_text: str) -> List[CodeBlock]:
    """
    Extract code blocks from markdown text with enhanced error handling

    Args:
        markdown_text: Markdown text to extract code blocks from

    Returns:
        List[CodeBlock]: List of extracted code blocks
    """
    if markdown_text is None:
        logging.info("[extract_markdown_code_blocks] Warning: markdown_text is None, returning empty list")
        return []

    if not isinstance(markdown_text, str):
        logging.info(f"[extract_markdown_code_blocks] Warning: markdown_text is not a string (type: {type(markdown_text)})")
        try:
            markdown_text = str(markdown_text)
        except Exception as e:
            logging.debug(f"[extract_markdown_code_blocks] Error: Failed to convert markdown_text to string: {e}")
            return []

    try:
        pattern = re.compile(r"```(?:\s*([\w\+\-]+))?\n([\s\S]*?)```")
        matches = pattern.findall(markdown_text)
        code_blocks: List[CodeBlock] = []
        for match in matches:
            language = match[0].strip() if match[0] else ""
            code_content = match[1]
            code_blocks.append(CodeBlock(code=code_content, language=language))
        return code_blocks
    except Exception as e:
        logging.debug(f"[extract_markdown_code_blocks] Error extracting code blocks: {e}")
        import traceback
        logging.debug(f"[extract_markdown_code_blocks] Traceback: {traceback.format_exc()}")
        return []





def clean_execution_output(output: str) -> str:
    """
    Clean execution output by removing noise patterns like pip installation messages,
    warnings, and other irrelevant information while preserving important diagnostic content.

    Args:
        output: Raw execution output string

    Returns:
        str: Cleaned output with noise removed
    """
    if not isinstance(output, str):
        return ""


    unwanted_block_pattern = r"^/workspace/tmp.*(?:\n^\s+.*)*\n?"
    cleaned_output = re.sub(unwanted_block_pattern, '', output, flags=re.MULTILINE)
    
    pip_noise_patterns = [
        r"Looking in indexes:.*\n?",
        r"Requirement already satisfied:.*\n?",
        r"Collecting .*\n?",
        r"Downloading .*\n?",
        r"Installing collected packages:.*\n?",
        r"Successfully installed.*\n?",
        r"WARNING: Running pip as.*\n?",
        r"\[notice\].*\n?",
    ]
    for pattern in pip_noise_patterns:
        cleaned_output = re.sub(pattern, '', cleaned_output)


    cleaned_output = re.sub(r'\n\s*\n', '\n', cleaned_output)
    final_output = cleaned_output.strip()

    return final_output


def truncate_output(output: str, max_output_length: int) -> tuple:
    """
    Truncate output results while preserving important information

    Args:
        output: Output string to truncate
        max_output_length: Maximum allowed length

    Returns:
        tuple: (truncated_output, is_truncated)
    """
    if output is None:
        logging.info("[truncate_output] Warning: output is None, returning empty string")
        return "", False

    if not isinstance(output, str):
        logging.info(f"[truncate_output] Warning: output is not a string (type: {type(output)}), converting")
        try:
            output = str(output)
        except Exception as e:
            logging.debug(f"[truncate_output] Error: Failed to convert output to string: {e}")
            return "", False

    if not output or len(output) <= max_output_length:
        return output, False

    half_length = max_output_length // 2
    start = output[:half_length]
    end = output[-half_length:]

    return f"{start}\n...\n[Output truncated, total {len(output)} characters]\n...\n{end}", True

def compress_duplicate_messages(output: str) -> str:
    """
    Compress duplicate warning and error messages
    """
    if not output:
        return output
        
    lines = output.split('\n')
    compressed_lines = []
    message_count = {}
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line in message_count:
            message_count[line] += 1
        else:
            message_count[line] = 1
            compressed_lines.append(line)
    
    final_lines = []
    for line in compressed_lines:
        count = message_count[line]
        if count > 3:
            final_lines.append(f"{line} (repeated {count} times)")
        else:
            final_lines.append(line)
            
    return '\n'.join(final_lines)

def check_anomaly_events_count(output: str, explorer_name: str) -> tuple:
    """
    Check if anomaly events count is within acceptable range

    Args:
        output: Execution result output
        explorer_name: Name of the explorer (e.g., "metric_explorer", "log_explorer", "trace_explorer")

    Returns:
        tuple: (is_valid, anomaly_count, error_message)
            - is_valid: True if count is within range, False otherwise
            - anomaly_count: Number of anomaly events found
            - error_message: Error message if count is invalid, None otherwise
    """
    # Extract anomaly events
    anomaly_events = extract_anomaly_events(output)
    anomaly_count = len(anomaly_events) if anomaly_events else 0

    # Determine thresholds based on explorer type
    if explorer_name.lower().startswith('log'):
        min_count = log_anomaly_events_min_count
        max_count = log_anomaly_events_max_count
        # Text length thresholds for fallback when no anomaly events detected
        min_text_length = 0
        max_text_length = 4000
    elif explorer_name.lower().startswith('metric'):
        min_count = metric_anomaly_events_min_count
        max_count = metric_anomaly_events_max_count
        # Text length thresholds for fallback when no anomaly events detected
        min_text_length = 500
        max_text_length = 10000
    elif explorer_name.lower().startswith('trace'):
        min_count = trace_anomaly_events_min_count
        max_count = trace_anomaly_events_max_count
        # Text length thresholds for fallback when no anomaly events detected
        min_text_length = 0
        max_text_length = 4000
    else:
        min_count = 0
        max_count = 1000
        # Default text length thresholds
        min_text_length = 0
        max_text_length = 10000
    if anomaly_count > 0:
        # Check if count is within range
        if anomaly_count > max_count:
            error_message = f"Too many anomaly events detected ({anomaly_count}), exceeding maximum allowed {max_count}"
            return False, anomaly_count, error_message
        elif anomaly_count < min_count:
            error_message = f"Too few anomaly events detected (only {anomaly_count}), below minimum expected {min_count}"
            return False, anomaly_count, error_message
        else:
            return True, anomaly_count, None
    else:
        # When no anomaly events are detected, fall back to validating by content length
        # Note: We do not have is_truncated flag here, so only length-based checks are applied
        try:
            content_length = len(''.join((output or '').split()))
        except Exception as e:
            # If any error occurs computing length, mark as invalid so caller can refine
            return False, anomaly_count, f"Failed to evaluate output length when no anomaly events detected: {e}"

        if content_length > max_text_length:
            error_message = (
                f"No anomaly events detected and output content too long ({content_length} characters), "
                f"max allowed: {max_text_length}"
            )
            return False, anomaly_count, error_message
        if content_length < min_text_length:
            error_message = (
                f"No anomaly events detected and output content too short ({content_length} characters), "
                f"min required: {min_text_length}"
            )
            return False, anomaly_count, error_message

        # Length is acceptable even with 0 events
        return True, anomaly_count, None



def extract_anomaly_events(output: str) -> list:
    """
    Extract anomaly events list from output results

    Args:
        output: Execution result output

    Returns:
        list: List containing anomaly event dictionaries
    """
    if output is None:
        logging.info("[extract_anomaly_events] Warning: output is None, returning empty list")
        return []

    if not isinstance(output, str):
        logging.info(f"[extract_anomaly_events] Warning: output is not a string (type: {type(output)}), converting to string")
        try:
            output = str(output)
        except Exception as e:
            logging.debug(f"[extract_anomaly_events] Error: Failed to convert output to string: {e}")
            return []

    if not output.strip():
        logging.info("[extract_anomaly_events] Warning: output is empty or whitespace only, returning empty list")
        return []

    # Find anomaly_events or events variable definition
    import re
    import json
    
    # First try to match JavaScript/Python style array definitions with a more lenient pattern
    js_array_pattern = r'(anomaly_events|events|anomalies)\s*=\s*\[([\s\S]*?)\]'
    js_match = re.search(js_array_pattern, output, re.DOTALL)
    
    if js_match:
        try:
            # Get array content
            array_content = js_match.group(2)
            
            # Separate each object item
            items = []
            brace_count = 0
            current_item = ""
            
            for char in array_content:
                if char == '{':
                    brace_count += 1
                    current_item += char
                elif char == '}':
                    brace_count -= 1
                    current_item += char
                    if brace_count == 0:
                        items.append(current_item.strip())
                        current_item = ""
                elif brace_count > 0:
                    current_item += char
            
            # Process each object item
            events = []
            for item in items:
                if not item:
                    continue
                
                # Manually parse JSON object to avoid JSON parsing issues
                try:
                    event = {}
                    # Remove braces
                    stripped = item.strip('{}')
                    # Split key-value pairs
                    for pair in re.split(r',\s*', stripped):
                        if ':' in pair:
                            key, value = pair.split(':', 1)
                            key = key.strip().strip('"\'')
                            value = value.strip().strip('"\'')
                            event[key] = value
                    if event:
                        events.append(event)
                except Exception:
                    pass
            
            if events:
                return events
        except Exception:
            pass
    
    # Try using a more robust method that doesn't rely on standard JSON parsing
    patterns = [
        r'anomaly_events\s*=\s*\[([\s\S]*?)\]',
        r'events\s*=\s*\[([\s\S]*?)\]',
        r'const\s+anomalies\s*=\s*\[([\s\S]*?)\]',
        r'anomaly_event\s*=\s*\[([\s\S]*?)\]',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, output, re.DOTALL)
        if match:
            events = []
            # Get entire array content
            array_content = match.group(1)
            
            # Use regex to match each object
            obj_matches = re.finditer(r'\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', array_content)
            for obj_match in obj_matches:
                try:
                    # Object content
                    obj_content = obj_match.group(1)
                    
                    # Manually extract key-value pairs
                    event = {}
                    for pair in re.finditer(r'(?:"?(\w+)"?\s*:|\'?(\w+)\'?\s*:)\s*(?:"([^"]*)"|(\'[^\']*\')|([^,}]*))', obj_content):
                        key = pair.group(1) or pair.group(2)
                        value = pair.group(3) or pair.group(4) or pair.group(5)
                        if value:
                            value = value.strip('\'"')
                        if key and value:
                            event[key] = value
                    
                    if event:
                        events.append(event)
                except Exception:
                    pass
            
            if events:
                return events
    
    # If still no valid anomaly events found, try directly matching well-formatted object lists
    obj_list_pattern = r'\[\s*(\{.*?\}(?:\s*,\s*\{.*?\})*)\s*\]'
    obj_list_match = re.search(obj_list_pattern, output, re.DOTALL)
    
    if obj_list_match:
        try:
            events = []
            list_content = obj_list_match.group(1)
            
            # Split each object
            obj_texts = re.findall(r'\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', list_content)
            
            for obj_text in obj_texts:
                event = {}
                # Extract key-value pairs
                for pair in re.finditer(r'(?:"?(\w+)"?\s*:|\'?(\w+)\'?\s*:)\s*(?:"([^"]*)"|(\'[^\']*\')|([^,}]*))', obj_text):
                    key = pair.group(1) or pair.group(2)
                    value = pair.group(3) or pair.group(4) or pair.group(5)
                    if value:
                        value = value.strip('\'"')
                    if key and value:
                        event[key] = value
                
                if event:
                    # Verify if this is an anomaly event object
                    if any(key in event for key in ['data_source', 'timestamp', 'description', 'cmdb_id']):
                        events.append(event)
            
            if events:
                return events
        except Exception:
            pass
    
    # If no valid anomaly events list found, return empty list
    return []

@default_subscription
class Executor(RoutedAgent):
    def __init__(self, code_executor: CodeExecutor) -> None:
        super().__init__("An executor agent.")
        self._code_executor = code_executor
        self.execution_result = None
        self.execution_code = None
        self._max_output_length = 5000  # Set maximum output length limit
        self._max_anomaly_events = 20  # Add maximum anomaly events limit
        self._refine_count = {}  # Add retry counter dictionary
        self._max_refine_attempts = 3  # Maximum retry attempts
        # temporary storage for refinement errors
        self._temp_errors = {}  # {coder_name: {error_type, error_message, failed_result, failed_code, refine_rules}}
        self._current_task = {}  # {coder_name: task_description}
        # store each coder's last non-empty result for fallback
        self._last_non_empty_results = {}  # {coder_name: truncated_output}
        Executor._instance = self

    def _extract_failed_code_from_blocks(self, code_blocks: List[CodeBlock]) -> str:
        failed_code = ""
        for block in code_blocks:
            if block.language.lower() in ['python', 'py']:
                failed_code += f"```python\n{block.code}\n```\n\n"
        return failed_code.strip()

    def _store_temp_error(self, coder_name: str, error_type: str, error_message: str,
                         failed_result: str, failed_code: str = "", refine_rules: str = None):
        """Store temporary error information"""
        self._temp_errors[coder_name] = {
            'error_type': error_type,
            'error_message': error_message,
            'failed_result': failed_result,
            'failed_code': failed_code,
            'refine_rules': refine_rules
        }
        logging.info(f"[CoderMemory] Temporarily stored {error_type} error for {coder_name}")

    def _is_empty_result(self, result: str) -> bool:
        if not result or not result.strip():
            return True

        stripped_result = result.strip()
        if stripped_result == "[]":
            return True

        import re
        no_whitespace = re.sub(r'\s+', '', stripped_result)
        if no_whitespace == "[]":
            return True

        if re.search(r'(anomaly_events|events|anomalies)\s*=\s*\[\s*\]', stripped_result):
            return True

        return False

    def _store_non_empty_result(self, coder_name: str, result: str):
        if not self._is_empty_result(result):
            self._last_non_empty_results[coder_name] = result
            logging.info(f"[Executor] Stored non-empty result for {coder_name}: {result[:100]}...")

    def _get_fallback_result(self, coder_name: str, current_result: str) -> str:
        if self._is_empty_result(current_result) and coder_name in self._last_non_empty_results:
            fallback_result = self._last_non_empty_results[coder_name]
            logging.info(f"[Executor] Current result is empty for {coder_name}, using fallback result: {fallback_result[:100]}...")
            return fallback_result
        return current_result

    def _extract_task_from_message(self, message_content: str) -> str:
        """Extract task description from message content"""
        import re

        # Enhanced robustness: handle None and non-string inputs
        if message_content is None:
            logging.info("[_extract_task_from_message] Warning: message_content is None")
            return "Unknown task"

        if not isinstance(message_content, str):
            logging.info(f"[_extract_task_from_message] Warning: message_content is not a string (type: {type(message_content)})")
            try:
                message_content = str(message_content)
            except Exception as e:
                logging.debug(f"[_extract_task_from_message] Error: Failed to convert message_content to string: {e}")
                return "Unknown task"

        try:
            # Try to extract task from <task> tags first
            task_match = re.search(r'<task>(.*?)</task>', message_content, re.DOTALL)
            if task_match:
                task_content = task_match.group(1).strip()
                return task_content  # Limit length

            # Fallback: extract content after coder name line
            lines = message_content.split('\n')
            found_coder_line = False
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Check if this is a coder name line (ends with :)
                if line.endswith(':') and ('_coder' in line or line in ['metric_coder', 'log_coder', 'trace_coder']):
                    found_coder_line = True
                    continue

                # If we found the coder line, return the next meaningful content
                if found_coder_line and line and not line.startswith('[') and 'From' not in line:
                    return line  # Limit length

            # If no task found, try to get the first meaningful line after skipping coder name
            for line in lines:
                line = line.strip()
                if (line and not line.startswith('[') and 'From' not in line and
                    not line.endswith(':') and '_coder' not in line):
                    return line  # Limit length

        except Exception as e:
            logging.debug(f"[_extract_task_from_message] Error processing message content: {e}")

        return "Unknown task"

    async def _save_successful_experience(self, coder_name: str, successful_result: str, 
                                        successful_code_blocks: list):
        if coder_name not in self._temp_errors or coder_name not in self._current_task:
            return
        
        temp_error = self._temp_errors[coder_name]
        task_description = self._current_task[coder_name]
        
        successful_solution = ""
        if successful_code_blocks:
            for block in successful_code_blocks:
                if block.language.lower() in ['python', 'py']:
                    successful_solution += f"```python\n{block.code}\n```\n\n"
        
        if not successful_solution:
            successful_solution = "Solution not captured in code blocks"
        
        try:
            start_time = datetime.now()
            await save_refinement_experience_to_memory(
                coder_name=coder_name,
                task_description=task_description,
                error_type=temp_error['error_type'],
                error_message=temp_error['error_message'],
                failed_execution_result=temp_error['failed_result'],
                successful_solution=successful_solution,
                successful_execution_result=successful_result,
                total_attempts=self._refine_count[coder_name],
                refine_rules=temp_error.get('refine_rules'),
                failed_code=temp_error.get('failed_code', '')
            )
            end_time = datetime.now()
            logging.info(f"[CoderMemory] Time used to save experience: {end_time - start_time}")
            logging.info(f"[CoderMemory] Memory saved: Coder: {coder_name}, Task: {task_description}, Error: {temp_error['error_type']}, Error Message: {temp_error['error_message']}, Attempts: {self._refine_count[coder_name]}, Successful Solution: {successful_solution[:100]}..., Successful Result: {successful_result[:100]}..., Refine Rules: {temp_error.get('refine_rules')}")
            logging.info(f"[CoderMemory] Saved complete refinement experience for {coder_name}: {temp_error['error_type']} -> Success")
        except Exception as e:
            logging.debug(f"[CoderMemory] Failed to save experience for {coder_name}: {e}")
        finally:
            del self._temp_errors[coder_name]
            self._refine_count[coder_name] = 0  

    @staticmethod
    def get_execution_result():
        """
        Get execution result and apply noise cleaning

        Returns:
            str: Cleaned execution result (never returns None)
        """
        if not Executor._instance:
            logging.info("[get_execution_result] Warning: No Executor instance found")
            return ""

        result = Executor._instance.execution_result
        if result is None:
            logging.info("[get_execution_result] Warning: execution_result is None")
            return ""

        if not isinstance(result, str):
            logging.info(f"[get_execution_result] Warning: execution_result is not a string (type: {type(result)}), converting")
            try:
                result = str(result)
            except Exception as e:
                logging.debug(f"[get_execution_result] Error: Failed to convert result to string: {e}")
                return ""

        return clean_execution_output(result)
    
    @staticmethod
    def get_execution_code():
        """
        Get executed code blocks
        
        Returns:
            List[CodeBlock]: Returns list of code blocks
        """
        if not Executor._instance or not Executor._instance.execution_code:
            return None
            
        if isinstance(Executor._instance.execution_code, list):
            return [block if isinstance(block, CodeBlock) else 
                   CodeBlock(code=block['code'], language=block['language']) 
                   for block in Executor._instance.execution_code]
        return None
    
    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        if message is None or message.content is None:
            logging.info("[Executor] Error: Received None message or message.content")
            return

        if not isinstance(message.content, str):
            logging.info(f"[Executor] Warning: message.content is not a string (type: {type(message.content)})")
            try:
                message_content = str(message.content)
            except Exception as e:
                logging.debug(f"[Executor] Error: Failed to convert message.content to string: {e}")
                return
        else:
            message_content = message.content

        try:
            coder_name = message_content.split(':\n')[0]
        except Exception as e:
            logging.debug(f"[Executor] Error extracting coder_name from message: {e}")
            coder_name = "unknown_coder"

        if coder_name not in self._refine_count:
            self._refine_count[coder_name] = 0

        task_description = self._extract_task_from_message(message_content)
        self._current_task[coder_name] = task_description
        
        code_blocks = extract_markdown_code_blocks(message_content)
        execute_code_blocks = []
        for code_block in code_blocks:
            if code_block.language.lower() in ['python', 'py']:
                execute_code_blocks.append(code_block)
                

        if execute_code_blocks:
            result = await self._code_executor.execute_code_blocks(
                execute_code_blocks, cancellation_token=ctx.cancellation_token
            )
            
            raw_output = result.output if result.output else ""
            compressed_output = clean_execution_output(raw_output)

            truncated_output,is_truncated = truncate_output(compressed_output,self._max_output_length)
            print(f"\n{'-'*80}\nExecutor truncated_output:\n{truncated_output}")
            
            if len(result.output) > 1 or str(result.output) == "[]":
                if self._refine_count[coder_name] >= self._max_refine_attempts:
                    final_result = self._get_fallback_result(coder_name, truncated_output)

                    logging.info(f"Maximum retry attempts ({self._max_refine_attempts}) reached, will use {'fallback' if final_result != truncated_output else 'current'} result. Execution result:\n" + "<success>"+final_result+"</success>")
                    await self.send_message(
                        Message(content=f"Maximum retry attempts ({self._max_refine_attempts}) reached, will use {'fallback' if final_result != truncated_output else 'current'} result. Execution result:\n" + "<success>"+final_result+"</success>"),
                        recipient=AgentId(coder_name, "default")
                    )
                    self.execution_result = final_result
                    if code_blocks[-1].language.lower() in ['python', 'py']:
                        self.execution_code = code_blocks
                    self._refine_count[coder_name] = 0 # Reset retry count
                    return

                if result.exit_code != 0:
                    self._refine_count[coder_name] += 1  # Increase retry count

                    failed_code = self._extract_failed_code_from_blocks(execute_code_blocks)

                    self._store_temp_error(coder_name, "execution_error",
                                         f"Code execution failed with exit code {result.exit_code}",
                                         truncated_output, failed_code)
                    
                    system_prompt = "When executing, code blocks will be executed sequentially, so if you need to install libraries, please install them in the first code block. Most standard Python environments do not support direct use of `!pip install` statements. You should avoid using this syntax and try to use subprocess to install required Python packages. If you are solving an error, you only need to provide the modified code. Please note that since all code blocks in your output will be executed to verify correctness, please ensure that the content in the output code blocks must be correct and executable. The execution failed with the following error:" + truncated_output
                    await self.send_message(Message(content=system_prompt),recipient=AgentId(coder_name, "default" ))
                else:
                    if truncated_output is None:
                        logging.info(f"[Executor] Warning: truncated_output is None for {coder_name}")
                        truncated_output = ""

                    # Use the reusable function to check anomaly events count
                    is_valid, anomaly_count, error_message = check_anomaly_events_count(truncated_output, coder_name)

                    logging.info(f"[Executor] Extracted {anomaly_count} anomaly events for {coder_name}, valid: {is_valid}")

                    # Select different refine_rules and thresholds based on different coders
                    if coder_name.lower().startswith('log'):
                        refine_rules = log_refine_rules
                        # Text length threshold (fallback mechanism)
                        min_length = 0
                        max_length = 2000
                    elif coder_name.lower().startswith('metric'):
                        refine_rules = metric_refine_rules
                        # Text length threshold (fallback mechanism)
                        min_length = 500
                        max_length = 10000
                    elif coder_name.lower().startswith('trace'):
                        refine_rules = trace_refine_rules
                        # Text length threshold (fallback mechanism)
                        min_length = 0
                        max_length = 2000
                    else:
                        refine_rules = "No specific refine rules defined for this coder."
                        min_length = 0
                        max_length = 10000

                    # If anomaly events are successfully extracted, use event count; otherwise fallback to text length
                    if anomaly_count > 0:
                        # Check if anomaly event count is appropriate using the reusable function
                        if not is_valid:
                            self._refine_count[coder_name] += 1  # Increase retry count

                            # Determine error type for storage
                            if "Too many" in error_message:
                                error_type = "too_many_events"
                                refine_message = f"{error_message}. Please increase detection threshold, focus only on the most severe anomalies, and consider temporal correlation of related anomalies, grouping related anomalies as single events. These are the refine principles:{refine_rules}.(Attempt {self._refine_count[coder_name]}/{self._max_refine_attempts}) Execution result:\n" + truncated_output
                            else:  # Too few events
                                error_type = "too_few_events"
                                refine_message = f"{error_message}. Please adjust code to discover more anomalies. These are the refine principles:{refine_rules}.(Attempt {self._refine_count[coder_name]}/{self._max_refine_attempts}) Execution result:\n" + truncated_output

                            failed_code = self._extract_failed_code_from_blocks(execute_code_blocks)

                            self._store_temp_error(coder_name, error_type, error_message, truncated_output, failed_code, refine_rules)

                            logging.info(refine_message)
                            await self.send_message(
                                Message(content=refine_message),
                                recipient=AgentId(coder_name, "default")
                            )
                            return
                        # Successful execution logic
                        self._store_non_empty_result(coder_name, truncated_output)

                        self.execution_result = truncated_output
                        if code_blocks[-1].language.lower() in ['python', 'py']:
                            self.execution_code = code_blocks

                        if coder_name in self._temp_errors:
                            await self._save_successful_experience(coder_name, truncated_output, code_blocks)

                        # Output different success messages based on whether anomaly events were detected
                        if anomaly_count > 0:
                            logging.info(f"Execution successful, detected {anomaly_count} anomaly events")
                        else:
                            logging.info(f"Execution successful, but no anomaly events detected in standard format")

                        await self.send_message(Message(content="<success>" + truncated_output + "</success>"),recipient=AgentId(coder_name, "default" ))
                    else:
                        # Failed to extract events, fallback to text length based judgment
                        content_length = len(''.join(truncated_output.split()))
                        
                        # Check if output length is appropriate
                        if content_length > max_length or is_truncated or content_length < min_length:
                            self._refine_count[coder_name] += 1  # Increase retry count
                            
                            if content_length > max_length or is_truncated:
                                failed_code = self._extract_failed_code_from_blocks(execute_code_blocks)

                                self._store_temp_error(coder_name, "content_length_too_long",
                                                     f"Too much output content ({content_length} characters), max allowed: {max_length}",
                                                     truncated_output, failed_code, refine_rules)
                                    
                                logging.info(f"Too much output content ({content_length} characters). Please increase detection threshold, focus only on the most severe anomalies, and ensure using standard format to return anomaly_events. These are the refine principles:{refine_rules}.(Attempt {self._refine_count[coder_name]}/{self._max_refine_attempts}) Execution result:\n" + truncated_output)
                                await self.send_message(
                                    Message(content=f"Too much output content ({content_length} characters). Please increase detection threshold, focus only on the most severe anomalies, and ensure using standard format to return anomaly_events. These are the refine principles:{refine_rules}.(Attempt {self._refine_count[coder_name]}/{self._max_refine_attempts}) Execution result:\n" + truncated_output),
                                    recipient=AgentId(coder_name, "default")
                                )
                            elif content_length < min_length:
                                failed_code = self._extract_failed_code_from_blocks(execute_code_blocks)

                                self._store_temp_error(coder_name, "content_length_too_short",
                                                     f"Too little output content ({content_length} characters), min required: {min_length}",
                                                     truncated_output, failed_code, refine_rules)
                                    
                                logging.info(f"Too little output content ({content_length} characters). Please adjust code to discover more anomalies, and ensure using standard format to return anomaly_events. These are the refine principles:{refine_rules}.(Attempt {self._refine_count[coder_name]}/{self._max_refine_attempts}) Execution result:\n" + truncated_output)
                                await self.send_message(
                                    Message(content=f"Too little output content ({content_length} characters). Please adjust code to discover more anomalies, and ensure using standard format to return anomaly_events. These are the refine principles:{refine_rules}.(Attempt {self._refine_count[coder_name]}/{self._max_refine_attempts}) Execution result:\n" + truncated_output),
                                    recipient=AgentId(coder_name, "default")
                                )
                            return

                        # Successful execution logic
                        self._store_non_empty_result(coder_name, truncated_output)

                        self.execution_result = truncated_output
                        if code_blocks[-1].language.lower() in ['python', 'py']:
                            self.execution_code = code_blocks

                        if coder_name in self._temp_errors:
                            await self._save_successful_experience(coder_name, truncated_output, code_blocks)

                        # Output different success messages based on whether anomaly events were detected
                        if anomaly_count > 0:
                            logging.info(f"Execution successful, detected {anomaly_count} anomaly events")
                        else:
                            logging.info(f"Execution successful, but no anomaly events detected in standard format")

                        await self.send_message(Message(content="<success>" + truncated_output + "</success>"),recipient=AgentId(coder_name, "default" ))
            else:
                self._store_non_empty_result(coder_name, truncated_output)

                self.execution_result = truncated_output
                self.execution_code = code_blocks
                         
@default_subscription
class MetricCoder(RoutedAgent):
    _instances = {}  # Use dictionary to store multiple instances
    # Add LLM call counter
    _llm_call_count = 0
    # Add token statistics
    _token_usage = {
        "prompt": 0,
        "completion": 0,
        "total": 0
    }
    
    def __init__(self, model_client: ChatCompletionClient, name: str = None) -> None:
        super().__init__("An assistant agent.")
        self._model_client = model_client
        self._name = name  # Store coder name
        self._chat_history: List[LLMMessage] = [
            SystemMessage(
                content=metric_system_coder,
            )
        ]
        MetricCoder._instances[name if name else id(self)] = self  # Use name or id as key to store instance

    @staticmethod
    def get_chat_history(instance_name=None):
        if instance_name and instance_name in MetricCoder._instances:
            return MetricCoder._instances[instance_name]._chat_history
        elif len(MetricCoder._instances) > 0:
            # If no name specified, return first instance's history
            return next(iter(MetricCoder._instances.values()))._chat_history
        return []
    
    @staticmethod
    def get_llm_call_count():
        """Get LLM call count"""
        return MetricCoder._llm_call_count
    
    @staticmethod
    def get_token_usage():
        """Get token usage statistics"""
        return MetricCoder._token_usage


    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        is_success = False
        success_match = re.search(r'<success>(.*?)</success>', message.content, re.DOTALL)
        if success_match:
            is_success = True
        
        enriched_content = message.content
        task_match = re.search(r'<task>(.*?)</task>', message.content, re.DOTALL)
        if task_match:
            task_content = task_match.group(1).strip()
            try:
                experience_memories = await retrieve_experience_memories(self._name, task_content, limit=3)
                if experience_memories:
                    memory_context = "\n\n=== Historical Refinement Experiences ===\n"
                    memory_context += "Based on similar tasks, here are some previous experiences to learn from:\n\n"
                    for i, memory in enumerate(experience_memories, 1):
                        memory_context += f"Experience {i}:\n{memory}\n\n"
                    memory_context += "=== End of Historical Experiences ===\n\n"
                    
                    enriched_content = enriched_content + memory_context
                    logging.info(f"[CoderMemory] Retrieved {len(experience_memories)} historical refinement experiences for {self._name}")
                else:
                    logging.info(f"[CoderMemory] No historical refinement experiences found for {self._name}")
            except Exception as e:
                logging.debug(f"[CoderMemory] Failed to retrieve experiences for {self._name}: {e}")
        
        self._chat_history.append(UserMessage(content=enriched_content, source="user"))
        
        if not is_success:
            # Increase LLM call count
            MetricCoder._llm_call_count += 1
            logging.info(f"[LLM Call Statistics] {self._name} called LLM, Total: {MetricCoder._llm_call_count}")
            
            result = await self._model_client.create(self._chat_history)
            
            # Check and update token usage statistics
            if hasattr(result, 'usage'):
                # RequestUsage format token statistics
                prompt_tokens = getattr(result.usage, 'prompt_tokens', 0)
                completion_tokens = getattr(result.usage, 'completion_tokens', 0)
                
                # Calculate total tokens
                total_tokens = prompt_tokens + completion_tokens
                
                MetricCoder._token_usage["prompt"] += prompt_tokens
                MetricCoder._token_usage["completion"] += completion_tokens
                MetricCoder._token_usage["total"] += total_tokens
                
                logging.info(f"[Token Statistics] {self._name}: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}, cumulative={MetricCoder._token_usage['total']}")
            
            print(f"\n{'-'*80}\n{self._name} Assistant:\n{result.content}")
            self._chat_history.append(AssistantMessage(content=result.content, source="assistant"))
            
            coder_message = Message(self._name + ":\n" + result.content)
            await self.send_message(
                coder_message,
                recipient=AgentId("executor", "default")
            )




@default_subscription
class LogCoder(RoutedAgent):
    _instances = {}  # Each Coder class uses its own instance dictionary
    # Add LLM call counter
    _llm_call_count = 0
    # Add token statistics
    _token_usage = {
        "prompt": 0,
        "completion": 0,
        "total": 0
    }
    
    def __init__(self, model_client: ChatCompletionClient, name: str = None) -> None:
        super().__init__("An assistant agent.")
        self._model_client = model_client
        self._name = name  # Store coder name
        self._chat_history: List[LLMMessage] = [
            SystemMessage(
                content=log_system_coder,
            )
        ]
        LogCoder._instances[name if name else id(self)] = self  # Use LogCoder's own dictionary

    @staticmethod
    def get_chat_history(instance_name=None):
        if instance_name and instance_name in LogCoder._instances:  # Use LogCoder's dictionary
            return LogCoder._instances[instance_name]._chat_history
        elif len(LogCoder._instances) > 0:
            # If no name specified, return first instance's history
            return next(iter(LogCoder._instances.values()))._chat_history
        return []
    
    @staticmethod
    def get_llm_call_count():
        """Get LLM call count"""
        return LogCoder._llm_call_count
    
    @staticmethod
    def get_token_usage():
        """Get token usage statistics"""
        return LogCoder._token_usage



    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        is_success = False
        success_match = re.search(r'<success>(.*?)</success>', message.content, re.DOTALL)
        if success_match:
            is_success = True
        
        # retrieve historical experiences
        enriched_content = message.content
        task_match = re.search(r'<task>(.*?)</task>', message.content, re.DOTALL)
        if task_match:
            task_content = task_match.group(1).strip()
            try:
                experience_memories = await retrieve_experience_memories(self._name, task_content, limit=3)
                if experience_memories:
                    memory_context = "\n\n=== Historical Refinement Experiences ===\n"
                    memory_context += "Based on similar tasks, here are some previous experiences to learn from:\n\n"
                    for i, memory in enumerate(experience_memories, 1):
                        memory_context += f"Experience {i}:\n{memory}\n\n"
                    memory_context += "=== End of Historical Experiences ===\n\n"
                    
                    enriched_content = enriched_content + memory_context
                    logging.info(f"[CoderMemory] Retrieved {len(experience_memories)} historical refinement experiences for {self._name}")
                else:
                    logging.info(f"[CoderMemory] No historical refinement experiences found for {self._name}")
            except Exception as e:
                logging.debug(f"[CoderMemory] Failed to retrieve experiences for {self._name}: {e}")
        
        self._chat_history.append(UserMessage(content=enriched_content, source="user"))

        
        if not is_success:
            # Increase LLM call count
            LogCoder._llm_call_count += 1
            logging.info(f"[LLM Call Statistics] {self._name} called LLM, Total: {LogCoder._llm_call_count}")
            
            result = await self._model_client.create(self._chat_history)
            
            # Check and update token usage statistics
            if hasattr(result, 'usage'):
                # RequestUsage format token statistics
                prompt_tokens = getattr(result.usage, 'prompt_tokens', 0)
                logging.info(f'result_usage:{result.usage}')
                completion_tokens = getattr(result.usage, 'completion_tokens', 0)
                
                # Calculate total tokens
                total_tokens = prompt_tokens + completion_tokens
                
                LogCoder._token_usage["prompt"] += prompt_tokens
                LogCoder._token_usage["completion"] += completion_tokens
                LogCoder._token_usage["total"] += total_tokens
                
                logging.info(f"[Token Statistics] {self._name}: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}, cumulative={LogCoder._token_usage['total']}")
            
            print(f"\n{'-'*80}\n{self._name} Assistant:\n{result.content}")
            self._chat_history.append(AssistantMessage(content=result.content, source="assistant"))
            
            coder_message = Message(self._name + ":\n" + result.content)
            await self.send_message(
                coder_message,
                recipient=AgentId("executor", "default")
            )


@default_subscription
class TraceCoder(RoutedAgent):
    _instances = {}  # Use dictionary to store multiple instances
    # Add LLM call counter
    _llm_call_count = 0
    # Add token statistics
    _token_usage = {
        "prompt": 0,
        "completion": 0,
        "total": 0
    }
    
    def __init__(self, model_client: ChatCompletionClient, name: str = None) -> None:
        super().__init__("An assistant agent.")
        self._model_client = model_client
        self._name = name  # Store coder name
        self._chat_history: List[LLMMessage] = [
            SystemMessage(
                content=trace_system_coder,
            )
        ]
        TraceCoder._instances[name if name else id(self)] = self  # Use name or id as key to store instance

    @staticmethod
    def get_chat_history(instance_name=None):
        if instance_name and instance_name in TraceCoder._instances:
            return TraceCoder._instances[instance_name]._chat_history
        elif len(TraceCoder._instances) > 0:
            # If no name specified, return first instance's history
            return next(iter(TraceCoder._instances.values()))._chat_history
        return []
    
    @staticmethod
    def get_llm_call_count():
        """Get LLM call count"""
        return TraceCoder._llm_call_count
    
    @staticmethod
    def get_token_usage():
        """Get token usage statistics"""
        return TraceCoder._token_usage

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        is_success = False
        success_match = re.search(r'<success>(.*?)</success>', message.content, re.DOTALL)
        if success_match:
            is_success = True
        
        # retrieve historical experiences
        enriched_content = message.content
        task_match = re.search(r'<task>(.*?)</task>', message.content, re.DOTALL)
        if task_match:
            task_content = task_match.group(1).strip()
            try:
                experience_memories = await retrieve_experience_memories(self._name, task_content, limit=3)
                if experience_memories:
                    memory_context = "\n\n=== Historical Refinement Experiences ===\n"
                    memory_context += "Based on similar tasks, here are some previous experiences to learn from:\n\n"
                    for i, memory in enumerate(experience_memories, 1):
                        memory_context += f"Experience {i}:\n{memory}\n\n"
                    memory_context += "=== End of Historical Experiences ===\n\n"
                    
                    enriched_content = enriched_content + memory_context
                    logging.info(f"[CoderMemory] Retrieved {len(experience_memories)} historical refinement experiences for {self._name}")
                    logging.info(f"[CoderMemory] experience_memories: {experience_memories}")
                else:
                    logging.info(f"[CoderMemory] No historical refinement experiences found for {self._name}")
            except Exception as e:
                logging.debug(f"[CoderMemory] Failed to retrieve experiences for {self._name}: {e}")
        
        self._chat_history.append(UserMessage(content=enriched_content, source="user"))
        
        if not is_success:
            # Increase LLM call count
            TraceCoder._llm_call_count += 1
            logging.info(f"[LLM Call Statistics] {self._name} called LLM, Total: {TraceCoder._llm_call_count}")
            
            result = await self._model_client.create(self._chat_history)
            
            # Check and update token usage statistics
            if hasattr(result, 'usage'):
                # RequestUsage format token statistics
                prompt_tokens = getattr(result.usage, 'prompt_tokens', 0)
                completion_tokens = getattr(result.usage, 'completion_tokens', 0)
                
                # Calculate total tokens
                total_tokens = prompt_tokens + completion_tokens
                
                TraceCoder._token_usage["prompt"] += prompt_tokens
                TraceCoder._token_usage["completion"] += completion_tokens
                TraceCoder._token_usage["total"] += total_tokens
                
                logging.info(f"[Token Statistics] {self._name}: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}, cumulative={TraceCoder._token_usage['total']}")
            
            print(f"\n{'-'*80}\n{self._name} Assistant:\n{result.content}")
            self._chat_history.append(AssistantMessage(content=result.content, source="assistant"))
            
            coder_message = Message(self._name + ":\n" + result.content)
            await self.send_message(
                coder_message,
                recipient=AgentId("executor", "default")
            )
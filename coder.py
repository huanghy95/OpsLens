import re
import logging
import os
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

# 添加自定义 CODER 日志级别
CODER_LEVEL = 25  # 介于 INFO (20) 和 WARNING (30) 之间
logging.addLevelName(CODER_LEVEL, "CODER")

def coder(self, message, *args, **kwargs):
    if self.isEnabledFor(CODER_LEVEL):
        self._log(CODER_LEVEL, message, args, **kwargs)

logging.Logger.coder = coder

logger = logging.getLogger(__name__)
logger.setLevel(CODER_LEVEL)
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

coder_handler = logging.FileHandler('coder.log', mode='w', encoding='utf-8')
coder_handler.setLevel(CODER_LEVEL)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
coder_handler.setFormatter(formatter)

logger.addHandler(coder_handler)
logger.propagate = False

from prompt import get_prompt_module
from NoteBook import NotebookSystem

prompt_module = get_prompt_module()

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
                content=log_anomaly_events_min_count,
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
        
        self._chat_history.append(UserMessage(content=enriched_content, source="user"))
        
        if not is_success:
            # Increase LLM call count
            Coder._llm_call_count += 1
            print(f"[LLM Call Statistics] {self._name} called LLM, Total: {Coder._llm_call_count}")
            
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
                
                print(f"[Token Statistics] {self._name}: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}, cumulative={Coder._token_usage['total']}")
            
            logger.coder(f"\n{'-'*80}\n{self._name} Assistant:\n{result.content}")
            self._chat_history.append(AssistantMessage(content=result.content, source="assistant"))
        
            compressed_output = compress_duplicate_messages(result.content)
            Coder._notebook.save_response(self._name, compressed_output)
            
            coder_message = Message(self._name + ":\n" + result.content)
        

            await self.send_message(
                coder_message,
                recipient=AgentId("executor", "default")
            )



def extract_markdown_code_blocks(markdown_text: str) -> List[CodeBlock]:
    pattern = re.compile(r"```(?:\s*([\w\+\-]+))?\n([\s\S]*?)```")
    matches = pattern.findall(markdown_text)
    code_blocks: List[CodeBlock] = []
    for match in matches:
        language = match[0].strip() if match[0] else ""
        code_content = match[1]
        code_blocks.append(CodeBlock(code=code_content, language=language))
    return code_blocks




def truncate_output(output: str, max_output_length: int) -> str:
    """
    Truncate output results while preserving important information
    """
    if not output or len(output) <= max_output_length:
        return output,False

    half_length = max_output_length // 2
    start = output[:half_length]
    end = output[-half_length:]
    
    return f"{start}\n...\n[Output truncated, total {len(output)} characters]\n...\n{end}",True

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

def extract_anomaly_events(output: str) -> list:
    """
    Extract anomaly events list from output results
    
    Args:
        output: Execution result output
    
    Returns:
        list: List containing anomaly event dictionaries
    """
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
        self._max_output_length = 50000  # Set maximum output length limit
        self._max_anomaly_events = 20  # Add maximum anomaly events limit
        self._refine_count = {}  # Add retry counter dictionary
        self._max_refine_attempts = 3  # Maximum retry attempts
        Executor._instance = self

    @staticmethod
    def get_execution_result():
        """
        Get execution result and remove pip installation and other noise information
        
        Returns:
            str: Cleaned execution result
        """
        if not Executor._instance:
            return None
            
        result = Executor._instance.execution_result
        if not result:
            return None
            
        # Noise patterns to filter
        noise_patterns = [
            r"Looking in indexes:.*",
            r"Requirement already satisfied:.*",
            r"WARNING: Running pip as.*",
            r"\[notice\].*",
            r"--------------------------------------------------------------------------------",
            r"Collecting .*",
            r"Downloading .*",
            r"Installing collected packages:.*",
            r"Successfully installed.*",
            r"━+.*",
            r"/workspace/tmp_code_.*",
            r"RuntimeWarning:.*",
            r"dt_utc = datetime.utcfromtimestamp(ts_seconds)\n",
        ]
        
        # Split by lines
        lines = result.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip empty lines
            if not line:
                continue
                
            # Check if it's a noise line
            is_noise = False
            for pattern in noise_patterns:
                if re.match(pattern, line):
                    is_noise = True
                    break
                    
            if not is_noise:
                cleaned_lines.append(line)
                
        return '\n'.join(cleaned_lines)
    
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
        coder_name = message.content.split(':\n')[0]  
        
        if coder_name not in self._refine_count:
            self._refine_count[coder_name] = 0
        
        code_blocks = extract_markdown_code_blocks(message.content)
        execute_code_blocks = []
        for code_block in code_blocks:
            if code_block.language.lower() in ['python', 'py']:
                execute_code_blocks.append(code_block)
                

        if execute_code_blocks:
            result = await self._code_executor.execute_code_blocks(
                execute_code_blocks, cancellation_token=ctx.cancellation_token
            )
            
            # compressed_output = compress_duplicate_messages(result.output)
            truncated_output,is_truncated = truncate_output(result.output,self._max_output_length)
            print(f"\n{'-'*80}\nExecutor:\n{truncated_output}")
            logger.coder(f"\n{'-'*80}\nExecutor:\n{truncated_output}")
            
            if len(result.output) > 5:
                if self._refine_count[coder_name] >= self._max_refine_attempts:
                    print(f"Maximum retry attempts ({self._max_refine_attempts}) reached, will use current result. Execution result:\n" + "<success>"+truncated_output+"</success>")
                    await self.send_message(
                        Message(content=f"Maximum retry attempts ({self._max_refine_attempts}) reached, will use current result. Execution result:\n" + "<success>"+truncated_output+"</success>"),
                        recipient=AgentId(coder_name, "default")
                    )
                    self.execution_result = truncated_output
                    if code_blocks[-1].language.lower() in ['python', 'py']:
                        self.execution_code = code_blocks
                    self._refine_count[coder_name] = 0 # Reset retry count
                    return

                if result.exit_code != 0:
                    self._refine_count[coder_name] += 1  # Increase retry count
                    system_prompt = "When executing, code blocks will be executed sequentially, so if you need to install libraries, please install them in the first code block. Most standard Python environments do not support direct use of `!pip install` statements. You should avoid using this syntax and try to use subprocess to install required Python packages. If you are solving an error, you only need to provide the modified code. Please note that since all code blocks in your output will be executed to verify correctness, please ensure that the content in the output code blocks must be correct and executable. The execution failed with the following error:" + truncated_output
                    await self.send_message(Message(content=system_prompt),recipient=AgentId(coder_name, "default" ))
                else:
                    # Extract anomaly events list
                    try:
                        anomaly_events = extract_anomaly_events(truncated_output)
                        anomaly_count = len(anomaly_events)
                    except Exception as e:
                        print(f"Failed to extract anomaly events: {e}")
                        anomaly_events = []
                        anomaly_count = 0
                    
                    # Select different refine_rules and thresholds based on different coders
                    if coder_name.lower().startswith('log'):
                        refine_rules = log_refine_rules
                        min_count = log_anomaly_events_min_count  # Should have at least 3 anomalies
                        max_count = log_anomaly_events_max_count  # No more than 15 anomalies
                        # Text length threshold (fallback mechanism)
                        min_length = 0
                        max_length = 600
                    elif coder_name.lower().startswith('metric'):
                        refine_rules = metric_refine_rules
                        min_count = metric_anomaly_events_min_count  # Should have at least 5 anomalies
                        max_count = metric_anomaly_events_max_count  # No more than 25 anomalies
                        # Text length threshold (fallback mechanism)
                        min_length = 500
                        max_length = 10000
                    elif coder_name.lower().startswith('trace'):
                        refine_rules = trace_refine_rules
                        min_count = trace_anomaly_events_min_count  # Should have at least 3 anomalies
                        max_count = trace_anomaly_events_max_count  # No more than 15 anomalies
                        # Text length threshold (fallback mechanism)
                        min_length = 0
                        max_length = 600
                    else:
                        refine_rules = "No specific refine rules defined for this coder."
                        min_count = 0
                        max_count = 1000
                        min_length = 0
                        max_length = 10000
                    # If anomaly events are successfully extracted, use event count; otherwise fallback to text length
                    if anomaly_count > 0:
                        # Check if anomaly event count is appropriate
                        if anomaly_count > max_count or is_truncated:
                            self._refine_count[coder_name] += 1  # Increase retry count
                            print(f"Too many anomaly events detected ({anomaly_count}), exceeding maximum allowed {max_count}. Please increase detection threshold, focus only on the most severe anomalies, and consider temporal correlation of related anomalies, grouping related anomalies as single events. These are the refine principles:{refine_rules}.(Attempt {self._refine_count[coder_name]}/{self._max_refine_attempts}) Execution result:\n" + truncated_output)
                            await self.send_message(
                                Message(content=f"Too many anomaly events detected ({anomaly_count}), exceeding maximum allowed {max_count}. Please increase detection threshold, focus only on the most severe anomalies, and consider temporal correlation of related anomalies, grouping related anomalies as single events. These are the refine principles:{refine_rules}.(Attempt {self._refine_count[coder_name]}/{self._max_refine_attempts}) Execution result:\n" + truncated_output),
                                recipient=AgentId(coder_name, "default")
                            )
                            return
                        elif anomaly_count < min_count:
                            self._refine_count[coder_name] += 1  # Increase retry count
                            print(f"Too few anomaly events detected (only {anomaly_count}), below minimum expected {min_count}. Please adjust code to discover more anomalies. These are the refine principles:{refine_rules}.(Attempt {self._refine_count[coder_name]}/{self._max_refine_attempts}) Execution result:\n" + truncated_output)
                            await self.send_message(
                                Message(content=f"Too few anomaly events detected (only {anomaly_count}), below minimum expected {min_count}. Please adjust code to discover more anomalies. These are the refine principles:{refine_rules}.(Attempt {self._refine_count[coder_name]}/{self._max_refine_attempts}) Execution result:\n" + truncated_output),
                                recipient=AgentId(coder_name, "default")
                            )
                            return
                        # Successful execution logic
                        self.execution_result = truncated_output
                        if code_blocks[-1].language.lower() in ['python', 'py']:
                            self.execution_code = code_blocks
                    else:
                        # Failed to extract events, fallback to text length based judgment
                        content_length = len(''.join(truncated_output.split()))
                        
                        # Check if output length is appropriate
                        if content_length > max_length or is_truncated or content_length < min_length:
                            self._refine_count[coder_name] += 1  # Increase retry count
                            
                            if content_length > max_length or is_truncated:
                                print(f"Too much output content ({content_length} characters). Please increase detection threshold, focus only on the most severe anomalies, and ensure using standard format to return anomaly_events. These are the refine principles:{refine_rules}.(Attempt {self._refine_count[coder_name]}/{self._max_refine_attempts}) Execution result:\n" + truncated_output)
                                await self.send_message(
                                    Message(content=f"Too much output content ({content_length} characters). Please increase detection threshold, focus only on the most severe anomalies, and ensure using standard format to return anomaly_events. These are the refine principles:{refine_rules}.(Attempt {self._refine_count[coder_name]}/{self._max_refine_attempts}) Execution result:\n" + truncated_output),
                                    recipient=AgentId(coder_name, "default")
                                )
                            elif content_length < min_length:
                                print(f"Too little output content ({content_length} characters). Please adjust code to discover more anomalies, and ensure using standard format to return anomaly_events. These are the refine principles:{refine_rules}.(Attempt {self._refine_count[coder_name]}/{self._max_refine_attempts}) Execution result:\n" + truncated_output)
                                await self.send_message(
                                    Message(content=f"Too little output content ({content_length} characters). Please adjust code to discover more anomalies, and ensure using standard format to return anomaly_events. These are the refine principles:{refine_rules}.(Attempt {self._refine_count[coder_name]}/{self._max_refine_attempts}) Execution result:\n" + truncated_output),
                                    recipient=AgentId(coder_name, "default")
                                )
                            return

                        # Successful execution logic
                        self.execution_result = truncated_output
                        if code_blocks[-1].language.lower() in ['python', 'py']:
                            self.execution_code = code_blocks
                        
                        # Output different success messages based on whether anomaly events were detected
                        if anomaly_count > 0:
                            print(f"Execution successful, detected {anomaly_count} anomaly events")
                        else:
                            print(f"Execution successful, but no anomaly events detected in standard format")
                            
                        await self.send_message(Message(content="<success>" + truncated_output + "</success>"),recipient=AgentId(coder_name, "default" ))
            else:
                self.execution_result = truncated_output
                self.execution_code = code_blocks
                
                
                
                
# BELOW IS UESLESS CODE      
                
                
                 
                
                
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
        
        self._chat_history.append(UserMessage(content=message.content, source="user"))
        
        if not is_success:
            # Increase LLM call count
            MetricCoder._llm_call_count += 1
            print(f"[LLM Call Statistics] {self._name} called LLM, Total: {MetricCoder._llm_call_count}")
            
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
                
                print(f"[Token Statistics] {self._name}: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}, cumulative={MetricCoder._token_usage['total']}")
            
            logger.coder(f"\n{'-'*80}\n{self._name} Assistant:\n{result.content}")
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
        
        self._chat_history.append(UserMessage(content=message.content, source="user"))

        
        if not is_success:
            # Increase LLM call count
            LogCoder._llm_call_count += 1
            print(f"[LLM Call Statistics] {self._name} called LLM, Total: {LogCoder._llm_call_count}")
            
            result = await self._model_client.create(self._chat_history)
            
            # Check and update token usage statistics
            if hasattr(result, 'usage'):
                # RequestUsage format token statistics
                prompt_tokens = getattr(result.usage, 'prompt_tokens', 0)
                print('result_usage',result.usage)
                completion_tokens = getattr(result.usage, 'completion_tokens', 0)
                
                # Calculate total tokens
                total_tokens = prompt_tokens + completion_tokens
                
                LogCoder._token_usage["prompt"] += prompt_tokens
                LogCoder._token_usage["completion"] += completion_tokens
                LogCoder._token_usage["total"] += total_tokens
                
                print(f"[Token Statistics] {self._name}: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}, cumulative={LogCoder._token_usage['total']}")
            
            logger.coder(f"\n{'-'*80}\n{self._name} Assistant:\n{result.content}")
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
        
        self._chat_history.append(UserMessage(content=message.content, source="user"))
        
        if not is_success:
            # Increase LLM call count
            TraceCoder._llm_call_count += 1
            print(f"[LLM Call Statistics] {self._name} called LLM, Total: {TraceCoder._llm_call_count}")
            
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
                
                print(f"[Token Statistics] {self._name}: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}, cumulative={TraceCoder._token_usage['total']}")
            
            logger.coder(f"\n{'-'*80}\n{self._name} Assistant:\n{result.content}")
            self._chat_history.append(AssistantMessage(content=result.content, source="assistant"))
            
            coder_message = Message(self._name + ":\n" + result.content)
            await self.send_message(
                coder_message,
                recipient=AgentId("executor", "default")
            )



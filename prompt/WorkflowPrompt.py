

planner_role_description = """
    You are a planner for a root cause analysis (RCA) framework called **OpsLens**. Your task is to generate a diagnosis plan for the **Investigating Stage** based on the provided context and the user's query. Here is an introduction to the framework:

    **OpsLens** is an LLM-based RCA solution designed to handle complex root cause analysis using multi-modal observability data from systems. It operates in three key stages:
    1. **Planning Stage**: OpsLens uses a reasoning-focused LLM to create a structured diagnosis plan from the user's RCA query. This plan offers high-level guidance for the investigation process and can be refined later if needed.
    2. **Investigating Stage**: A multi-agent system conducts the investigation. A controller agent follows the diagnosis plan and delegates tasks to three specialized explorer agents — one each for **trace**, **log**, and **metric** data. Please note that the focus of the investigation is on metrics; do not rely excessively on logs and traces.
    * Each explorer maintains long-term memory for its own data type and shares short-term memory for coordination.
    * Instead of directly analyzing raw data, explorers **generate diagnostic tools** (i.e., executable code) using a two-step approach: **task analysis** followed by **code generation**, with feedback-driven refinement to ensure tool quality.
    * These tools are saved in a **Generated Toolset** and executed to produce **diagnosis events**.
    3. **Reasoning Stage**: Once the controller deems the investigation sufficient, OpsLens moves to the reasoning stage. A reasoning-enhanced LLM, supported by heuristic rules mined during investigation, analyzes the diagnosis events, infers the root cause, and responds to the user.

    Your role is to create a **diagnosis plan** that aligns with this workflow and supports efficient and accurate investigation.
"""


planning_msg_template = """
<role_description>
{planner_role_description}
</role_description>
<user_query>
{user_query}
</user_query>
<data_source_information>
{background}
</data_source_information>
Please generate a diagnosis plan.
"""

investigation_msg_template = """
<user query>
{user_query}
</user query>
<diagnosis plan>
{diagnosis_plan}
</diagnosis plan>

please decide:
1. what should be investigated first?
2. which explorer should be called?

please return the decision in the following format:
{{"explorer": "explorer name", "task": "specific investigation task"}}
"""


investigator_prompt_template = """
{investigator_msg}

available explorers:
{available_explorers}

please decide which explorer to use for investigation. the format must be:
{{"explorer": "explorer name", "task": "specific investigation task, it should include the specific time range of the event"}}

If you think the investigation is complete, please reply INVESTIGATION_COMPLETE
"""


exception_investigator_prompt_template = """
the format of the last response is incorrect or the explorer is invalid. 

Available explorers are: 
{available_explorers}

Please only return one of the following formats:
1. continue investigation: {{"explorer": "explorer name,it should be one of the available explorers", "task": "specific investigation task,it should based on the plan step by step."}}
2. complete investigation: INVESTIGATION_COMPLETE

Important: You must perform only one of these two actions.do NOT include any other explanatory text.
"""


update_investigator_prompt_template = """
<completed investigation>
{investigation_results}
</completed investigation>

Please check the completed investigation results and decide:

1. Whether all necessary information has been collected. If all information has been collected, please reply: INVESTIGATION_COMPLETE.

2. If there is missing information, please return the corresponding investigation task. Please note that not every modality can provide additional information for the RCA task. If the return result is empty, it is very likely that the corresponding information cannot be found:
    {{"explorer": "explorer name", "task": "specific investigation task"}}
"""





reasoning_msg_template = """
<user query>
{user_query}
</user query>
<failure information>
{queried_issue}
</failure information>
<diagnosis plan>
{diagnosis_plan}
</diagnosis plan>
investigation results: {diagnosis_events}

please analyze the root cause and give suggestions.
"""








explorer_task_prompt = """
<investigate task>
{task}
</investigate task>

You are an explorer with tool execution capabilities. Use a ReAct-style loop to complete the task.

At each turn, you must do exactly one of the following:

1. Tool Execution: If a suitable tool exists in tool_list, select and execute it.
   - If further steps are required after execution, return exactly: CONTINUE
   - If the task is fully completed, return exactly two lines:
     TASK_COMPLETE
     anomaly_event = [ ... ]  # See the format below

2. Request Tool Generation: If no appropriate tool is available, return exactly: NEED_TOOL_GENERATION

Important rules:
- Keep responses minimal and only use the specified control tokens (CONTINUE, NEED_TOOL_GENERATION, TASK_COMPLETE).
- Only include the anomaly_event block when returning TASK_COMPLETE.
"""


tool_execution_result_prompt = """
<task description>
{task}
</task description>

<tool's execution results>
{generated_tool_execution_result}
</tool's execution results>

Please summarize a straightforward answer to the question based on the execution results. Use plain English.

Provide a detailed analysis of the code execution result from Executor in the last step, with detailed reasoning of 'what have been done' and 'what can be derived' prepare for the next step in root cause analysis. 

Respond in the following format:

anomaly_event = [
    {{
        data_source: "Trace",  # Enum type, possible values: ("Metric", "Log", "Trace")
        timestamp: "2021-03-04 14:57:00", 
        cmdb_id: "Tomcat01",(optional,only if you can get the cmdb_id from the tool's execution results)
        description: "' is abnormally high,delta=...", (optional,only if you can get the description from the tool's execution results)
    }}
]   

Note: If the tool's execution results based on this modality show no detected anomalies, simply return `anomaly_event = []`. Do not create any new anomaly events.
"""


tool_generation_format = """
**Tool Generation Instructions**

You are a Python tool generation expert. Please strictly follow the rules below to generate a Python tool function for root cause analysis. Note that all the code logic you generate must be contained within a single main function, as the code you produce will be saved to a code repository for direct invocation later. 

1. **Single function entry**: 
   * All code logic must be encapsulated in one function, including all hyperparameters. And function name should highly summarize the task.
   * DO NOT GENERATE MORE THAN ONE FUNCTION IN ONE TASK.
   * The code you generate needs to run immediately to produce results, so you can use "if __name__ == '__main__'" to execute it.
   * Do not print useless content or intermediate results in the function; only the final anomaly_events needs to be printed. If the program includes a main function (or a main execution logic entry point), the printing behavior must also be controlled.

2. **Function Signature**:
   * Use `typing.Annotated` to add type hints and clear natural language descriptions for **all** parameters. Include example formats (e.g., showing the expected format for string parameters) to facilitate future usage.
   * Place core parameters that define the task (such as `start_time`, `target_kpis`) at the front. These define "what" the task does and should not have default values.
   * Place hyperparameters used to fine-tune algorithms (such as `threshold`, `window_size`) later. These define "how" the task is done and should have **reasonable default values**.

3. **Docstring**:
   * Must include a `__doc__` string.
   * The first line is a brief summary of the tool's functionality.
   * Subsequent content should explain in detail what the tool does, its working principle, and its output.

4. **File Paths and Data Structure**:
   * **Do not hardcode any absolute or relative file paths for the root directory.**
   * For this specific generation task, you should assume the data is located under the root path data_source_path=`{Data_Relative_Root_Path}`.
   * **Your generated tool must assume a specific data directory structure under this path**: `[data_source_path]/[Date]/[Type]/[FileName]`(Only ).
       * `[Date]`: This is the **only** variable part of the path that should be exposed as a **function parameter**. For example, a parameter `date_str: Annotated[str, "Date string in 'YYYY_MM_DD' format, e.g., '2021_03_12'"]`.
       * `[Type]` (e.g., 'metric', 'trace', 'log') and `[FileName]` (e.g., 'metric_container.csv') are considered fixed for a given task. You should construct the full path to the required file(s) *inside* the function logic, not expose them as parameters.
       * **Example of path construction inside the function**: `metric_path = os.path.join(data_source_path, date_str, 'metric', 'metric_container.csv')`.
       * Do NOT use data_source_path as a function parameter. Since it is a fixed value, we emphasize again that only [Date] can be used as a function parameter. Later, the complete path will be concatenated inside the function.

5. **Return Value**:
   * The primary output of the function should be a anomaly_events defined as below. Do not limit the number of returned anomaly_events. However, it is necessary to ensure that the number of characters in the returned anomaly_events does not exceed 7,000.
   * Avoid using only `print()` for output results. Do not use `print` for logging or status updates.
   * Do not use print to output intermediate results; only print the final anomaly_events. To emphasize again, do not print the results of intermediate processes; only print and return the final answer.

6. **Dependencies**: All required libraries (such as `pandas`, `numpy`) should be imported at the top of the file.

7. **Error Handling**:
   - Always raise errors instead of except. When an agent invokes a tool, if the tool's code does not terminate due to an error, it will only return the final result from the return statement. Therefore, it is necessary to raise an error or allow the program to fail naturally, rather than using a try-except block.
   - Return an empty list for empty results
   - Standardize error messages

8. Please strictly follow the following rules for parameter definition:
   - Only basic data types are allowed for function parameters (e.g., int, str, bool, float, lists, dictionaries, etc.). Complex data types (such as pd.Series, custom class instances, etc.) are prohibited.
   - All function parameters must be wrapped with the Annotated type (the Annotated declaration cannot be omitted).
   - Hyperparameters related to the function can only be defined in the function's parameter list. They are prohibited from being placed in locations other than the function parameters (e.g., global variables, non-parameter variables inside the function, etc.).

```python
from typing import Annotated, Any, Optional
from datetime import datetime

def generated_tool_name(
    param1: Annotated[str, "Description of the first parameter"],
    param2: Annotated[int, "Description of the second parameter"],
    param3: Annotated[Optional[datetime], "An optional datetime parameter"] = None,
    param4: Annotated[Any, "A more generic parameter, which can accept any type"] = None
) -> RETURNVALUE:
    \"\"\"
    Tool description:
    Provide a concise and clear description of what the tool does.
    
    Example:
    This tool analyzes the KPIs (CPU, memory, disk, network) for components in a given time window to detect any abnormal resource usage.

    Parameters:
    - param1: Description of the first parameter (e.g., "CPU_usage" or "yyyy-mm-dd")
    - param2: Description of the second parameter (e.g., 1045 for server ID or 5 for network interface)
    - param3: An optional datetime parameter for specifying a custom time range 
              (e.g., datetime(2023, 10, 1, 8, 30) or None for default range)
    - param4: An additional parameter that can be used to specify extra options
              (e.g., {{"threshold": 90, "alert": True}} or "verbose" for detailed output)

    Returns:
    - RETURNVALUE (the function may perform side-effects like writing to a database or printing results).
    \"\"\"
    
    # Your tool's logic here
    # Ensure this logic is generalized for easy reuse across tasks
    
if __name__ == '__main__':
	# Use the tool here
"""





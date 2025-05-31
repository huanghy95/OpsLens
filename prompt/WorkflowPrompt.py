

planner_role_description = """
    You are a planner for a root cause analysis (RCA) framework called **OpsLens**. Your task is to generate a diagnosis plan for the **Investigating Stage** based on the provided context and the user's query. Here is an introduction to the framework:

    **OpsLens** is an LLM-based RCA solution designed to handle complex root cause analysis using multi-modal observability data from systems. It operates in three key stages:
    1. **Planning Stage**: OpsLens uses a reasoning-focused LLM to create a structured diagnosis plan from the user's RCA query. This plan offers high-level guidance for the investigation process and can be refined later if needed.
    2. **Investigating Stage**: A multi-agent system conducts the investigation. A controller agent follows the diagnosis plan and delegates tasks to three specialized explorer agents — one each for **trace**, **log**, and **metric** data.
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

if you think the investigation is complete, please reply INVESTIGATION_COMPLETE
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

1. Whether all necessary information has been collected, if a tool call fails, please try to call it again based on the failure information.

    - System resource usage


2. If there is missing information, please return the corresponding investigation task:
    {{"explorer": "explorer name", "task": "specific investigation task"}}

3. If all information has been collected, please reply: INVESTIGATION_COMPLETE
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

As an explorer with tool execution capabilities, your first step is to check whether the given tool_list contains a tool that can complete the investigation task. Based on your assessment, you must take one of two actions only:  

1. Tool Execution: If a suitable tool exists in tool_list, select and execute it.  
2. Request Tool Generation: If no appropriate tool is available, return: "NEED_TOOL_GENERATION". 

Important: You must perform only one of these two actions and produce no additional output.
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








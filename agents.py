import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="flaml")
import asyncio

from dataclasses import dataclass

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from memory import load_memory, planer_memory, investigator_memory, log_explorer_memory, metric_explorer_memory, trace_explorer_memory, reasoner_memory

from prompt import get_prompt_module

prompt_module = get_prompt_module()

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

async def initialize_memory():
    await load_memory()

# 
if __name__ == "__main__":
    asyncio.run(initialize_memory())



reason_model_client =  OpenAIChatCompletionClient(
        model="",
        base_url="",
        api_key="",
        model_info={
        "vision": True,
        "function_calling": True,
        "json_output": False,
        "family": "unknown",
        },
        temperature=0,
)

model_client =  OpenAIChatCompletionClient(
        model="",
        base_url="",
        api_key="",
        model_info={
        "vision": True,
        "function_calling": True,
        "json_output": False,
        "family": "unknown",
        },
        temperature=0,
)




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
)


log_explorer = AssistantAgent(
    name="log_explorer",
    description="Agent for exploring log data",
    system_message="You are the log explorer "+system_explorer,
    model_client=model_client,
)


trace_explorer = AssistantAgent(
    name="trace_explorer",
    description="Agent for exploring trace data",
    system_message="You are the trace explorer "+system_explorer,
    model_client=model_client,
)


# Reasoner Agent
reasoner_agent = AssistantAgent(
    name="reasoner",
    system_message=system_reasoner,
    model_client=model_client,
)





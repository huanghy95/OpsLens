import json
from pathlib import Path
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
import asyncio

def load_metadata():
    metadata_path = Path(__file__).parent / "memory.json"
    with open(metadata_path, 'r', encoding='utf-8') as f:
        return json.load(f)

METADATA = load_metadata()

planer_memory = ListMemory()
investigator_memory = ListMemory()
log_explorer_memory = ListMemory()
metric_explorer_memory = ListMemory()
trace_explorer_memory = ListMemory()
coder_memory = ListMemory()
reasoner_memory = ListMemory()


async def add_planer_memory():
    content = json.dumps(METADATA["planer"], ensure_ascii=False)
    await planer_memory.add(
        MemoryContent(
            content=content,
            mime_type=MemoryMimeType.TEXT,
        )
    )


async def add_investigator_memory():
    content = json.dumps(METADATA["investigator"], ensure_ascii=False)
    await investigator_memory.add(
        MemoryContent(
            content=content,
            mime_type=MemoryMimeType.TEXT,
        )
    )


async def add_log_explorer_memory():
    content = json.dumps(METADATA["log"], ensure_ascii=False)
    await log_explorer_memory.add(
        MemoryContent(
            content=content,
            mime_type=MemoryMimeType.TEXT,
        )
    )


async def add_metric_explorer_memory():
    content = json.dumps(METADATA["metric"], ensure_ascii=False)
    await metric_explorer_memory.add(
        MemoryContent(
            content=content,
            mime_type=MemoryMimeType.TEXT,
        )
    )


async def add_trace_explorer_memory():
    content = json.dumps(METADATA["trace"], ensure_ascii=False)
    await trace_explorer_memory.add(
        MemoryContent(
            content=content,
            mime_type=MemoryMimeType.TEXT,
        )
    )




async def add_reasoner_memory():
    content = json.dumps(METADATA["reasoner"], ensure_ascii=False)
    await reasoner_memory.add(
        MemoryContent(
            content=content,
            mime_type=MemoryMimeType.TEXT,
        )
    )



async def load_memory():
    await add_planer_memory()
    await add_investigator_memory()
    await add_log_explorer_memory()
    await add_metric_explorer_memory()
    await add_trace_explorer_memory()
    await add_reasoner_memory()



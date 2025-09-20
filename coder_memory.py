import asyncio
import json
import traceback
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import chromadb
import openai
from config_manager import get_coder_memory_paths, get_openai_config
import logging

@dataclass
class CodeExperience:
    """Data structure for storing code experience (including comparison analysis of failed and successful code)"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    coder_name: str = ""  # "metric_coder", "log_coder", "trace_coder"
    task_description: str = ""  # Task description

    # Failed code related
    failed_code: str = ""  # Failed code
    failed_execution_result: str = ""  # Failed execution result
    error_type: str = ""  # Error type
    error_message: str = ""  # Error message

    # Successful code related
    successful_code: str = ""  # Successful code
    successful_execution_result: str = ""  # Successful execution result

    # LLM analysis summary
    llm_analysis: str = ""  # LLM comparison analysis result
    coding_rules: str = ""  # LLM summarized coding rules

    # Metadata
    total_attempts: int = 1
    created_at: datetime = field(default_factory=datetime.utcnow)
    tags: List[str] = field(default_factory=list)


class ChromaMemoryManager:
    """Chroma-based memory manager that maintains independent vector databases for each modality data"""

    def __init__(self, storage_dir: str = None):
        """
        Initialize memory manager

        Args:
            storage_dir: Storage directory path, if None will use config manager
        """
        if storage_dir is None:
            memory_paths = get_coder_memory_paths()
            storage_dir = memory_paths["chroma_storage"]

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Chroma client
        self.chroma_client = chromadb.PersistentClient(path=str(self.storage_dir / "chroma_db"))

        # Create independent collections for each modality data
        self.collections = {}
        self.coder_types = ["metric_coder", "log_coder", "trace_coder"]

        for coder_type in self.coder_types:
            collection_name = f"{coder_type}_experiences"
            self.collections[coder_type] = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )

        # OpenAI configuration (for embedding and LLM analysis)
        self.openai_config = get_openai_config()

        self._initialized = True
        logging.info(f"[ChromaMemoryManager] Initialized with collections: {list(self.collections.keys())}")
    
    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate text embedding using OpenAI's text-embedding-3-large model

        Args:
            text: Text to generate embedding for

        Returns:
            List[float]: embedding vector
        """
        try:
            client = openai.OpenAI(
                api_key=self.openai_config["api_key"],
                base_url=self.openai_config["base_url"]
            )

            # Preprocess text
            processed_text = text.replace("\n", " ").strip()
            if not processed_text:
                return []

            response = client.embeddings.create(
                model="text-embedding-3-large",
                input=[processed_text]
            )

            return response.data[0].embedding

        except Exception as e:
            logging.debug(f"[ChromaMemoryManager] Error generating embedding: {e}")
            logging.debug(traceback.format_exc())
            return []
    
    async def _analyze_code_difference(self, failed_code: str, successful_code: str, task_description: str) -> Dict[str, str]:
        """
        Use LLM to analyze differences between failed and successful code, generate experience summary

        Args:
            failed_code: Failed code
            successful_code: Successful code
            task_description: Task description

        Returns:
            Dict[str, str]: Dictionary containing 'analysis' and 'rules'
        """
        prompt = f"""You are a code analysis expert. I will provide you with 2 code segments: the first is historically failed code, and the second is code that successfully completed the task. I need you to analyze the differences in detail and summarize coding rules for future similar tasks.

Task Description:
{task_description}

Failed Code:
```python
{failed_code}
```

Successful Code:
```python
{successful_code}
```

Please respond in the following format:

## Code Difference Analysis
[Detailed analysis of key differences between the two code segments, including logic, implementation approach, error handling, etc.]

## Coding Rules Summary
[Based on the difference analysis, summarize specific coding rules and best practices for future similar tasks]
"""
        
        try:
            client = openai.OpenAI(
                api_key=self.openai_config["api_key"],
                base_url=self.openai_config["base_url"]
            )
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content

            # Parse analysis results
            analysis_section = ""
            rules_section = ""

            if "## Code Difference Analysis" in content:
                parts = content.split("## Coding Rules Summary")
                analysis_section = parts[0].replace("## Code Difference Analysis", "").strip()
                if len(parts) > 1:
                    rules_section = parts[1].strip()
            else:
                analysis_section = content

            return {
                "analysis": analysis_section,
                "rules": rules_section
            }

        except Exception as e:
            logging.debug(f"[ChromaMemoryManager] Error analyzing code difference: {e}")
            logging.debug(traceback.format_exc())
            return {
                "analysis": "Analysis failed",
                "rules": "Unable to generate rules"
            }
    
    async def save_experience(self, experience: CodeExperience) -> bool:
        """
        Save code experience to vector database

        Args:
            experience: CodeExperience object

        Returns:
            bool: Whether save was successful
        """
        if experience.coder_name not in self.collections:
            logging.info(f"[ChromaMemoryManager] Unknown coder type: {experience.coder_name}")
            return False

        try:
            # If no LLM analysis exists, perform analysis first
            if not experience.llm_analysis and experience.failed_code and experience.successful_code:
                analysis_result = await self._analyze_code_difference(
                    experience.failed_code,
                    experience.successful_code,
                    experience.task_description
                )
                experience.llm_analysis = analysis_result["analysis"]
                experience.coding_rules = analysis_result["rules"]

            # Generate text for retrieval (task description + coding rules)
            search_text = f"{experience.task_description} {experience.coding_rules}".strip()
            embedding = self._generate_embedding(search_text)

            if not embedding:
                logging.info(f"[ChromaMemoryManager] Failed to generate embedding for experience {experience.id}")
                return False

            # Build document content
            document_content = self._format_experience_document(experience)

            # Build metadata
            metadata = {
                "experience_id": experience.id,
                "coder_name": experience.coder_name,
                "task_description": experience.task_description,
                "error_type": experience.error_type,
                "total_attempts": experience.total_attempts,
                "created_at": experience.created_at.isoformat(),
                "tags": ",".join(experience.tags)
            }

            # Save to Chroma
            collection = self.collections[experience.coder_name]
            collection.add(
                embeddings=[embedding],
                documents=[document_content],
                metadatas=[metadata],
                ids=[experience.id]
            )

            logging.info(f"[ChromaMemoryManager] Saved experience {experience.id} for {experience.coder_name}")
            return True

        except Exception as e:
            logging.debug(f"[ChromaMemoryManager] Error saving experience: {e}")
            logging.debug(traceback.format_exc())
            return False
    
    def _format_experience_document(self, experience: CodeExperience) -> str:
        """Format experience as document content"""
        return f"""
Code Experience Record:
- Coder: {experience.coder_name}
- Task: {experience.task_description}
- Created: {experience.created_at.strftime('%Y-%m-%d %H:%M:%S')}

Problem Analysis:
- Error Type: {experience.error_type}
- Error Message: {experience.error_message}
- Attempts: {experience.total_attempts}

Failed Code:
{experience.failed_code[:500]}...

Successful Code:
{experience.successful_code[:500]}...

LLM Difference Analysis:
{experience.llm_analysis}

Coding Rules Summary:
{experience.coding_rules}

Tags: {', '.join(experience.tags)}
        """.strip()
    
    async def retrieve_similar_experiences(self, coder_name: str, task_description: str, limit: int = 5) -> List[str]:
        """
        Retrieve historical experiences similar to current task

        Args:
            coder_name: Coder name
            task_description: Current task description
            limit: Limit of retrieval results

        Returns:
            List[str]: List of relevant experience memories
        """
        if coder_name not in self.collections:
            return []

        try:
            # Generate query vector
            query_embedding = self._generate_embedding(task_description)
            if not query_embedding:
                return []

            # Retrieve from corresponding collection
            collection = self.collections[coder_name]
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=limit
            )

            if not results['documents'] or not results['documents'][0]:
                return []

            logging.info(f"[ChromaMemoryManager] Retrieved {len(results['documents'][0])} experiences for {coder_name}")
            return results['documents'][0]

        except Exception as e:
            logging.debug(f"[ChromaMemoryManager] Error retrieving experiences: {e}")
            logging.debug(traceback.format_exc())
            return []


# Global memory manager instance
_global_memory_manager: Optional[ChromaMemoryManager] = None


async def get_memory_manager() -> ChromaMemoryManager:
    """Get global memory manager instance"""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = ChromaMemoryManager()
    return _global_memory_manager


async def save_refinement_experience_to_memory(
    coder_name: str,
    task_description: str,
    error_type: str,
    error_message: str,
    failed_execution_result: str,
    successful_solution: str,
    successful_execution_result: str,
    total_attempts: int,
    refine_rules: Optional[str] = None,
    failed_code: str = ""
) -> bool:
    """
    Convenience function: Save refinement experience to memory (failed code + successful code comparison analysis)

    Args:
        coder_name: Coder name
        task_description: Task description
        error_type: Error type
        error_message: Error message
        failed_execution_result: Failed execution result
        successful_solution: Successful solution (code)
        successful_execution_result: Successful execution result
        total_attempts: Total attempts
        refine_rules: Refinement rules (optional)
        failed_code: Failed code (optional)

    Returns:
        bool: Whether save was successful
    """
    memory_manager = await get_memory_manager()

    # Create CodeExperience object
    experience = CodeExperience(
        coder_name=coder_name,
        task_description=task_description,
        failed_code=failed_code if failed_code else failed_execution_result,  # Use failed_code if provided, otherwise fallback to failed_execution_result
        failed_execution_result=failed_execution_result,
        error_type=error_type,
        error_message=error_message,
        successful_code=successful_solution,
        successful_execution_result=successful_execution_result,
        total_attempts=total_attempts,
        tags=[error_type] if error_type else []
    )

    # If refine_rules exists, add to tags
    if refine_rules:
        experience.tags.append("refined")

    return await memory_manager.save_experience(experience)


async def retrieve_experience_memories(coder_name: str, current_task: str, limit: int = 3) -> List[str]:
    """
    Convenience function: Retrieve refinement experience memories

    Args:
        coder_name: Coder name
        current_task: Current task description
        limit: Retrieval limit

    Returns:
        List[str]: List of relevant experience memories
    """
    memory_manager = await get_memory_manager()
    return await memory_manager.retrieve_similar_experiences(coder_name, current_task, limit)


async def save_code_comparison_experience(
    coder_name: str,
    task_description: str,
    failed_code: str,
    successful_code: str,
    error_type: str = "",
    error_message: str = "",
    failed_execution_result: str = "",
    successful_execution_result: str = "",
    total_attempts: int = 1,
    tags: Optional[List[str]] = None
) -> bool:
    """
    New convenience function: Directly save comparison experience of failed and successful code

    Args:
        coder_name: Coder name
        task_description: Task description
        failed_code: Failed code
        successful_code: Successful code
        error_type: Error type (optional)
        error_message: Error message (optional)
        failed_execution_result: Failed execution result (optional)
        successful_execution_result: Successful execution result (optional)
        total_attempts: Total attempts
        tags: Tag list (optional)

    Returns:
        bool: Whether save was successful
    """
    memory_manager = await get_memory_manager()

    experience = CodeExperience(
        coder_name=coder_name,
        task_description=task_description,
        failed_code=failed_code,
        failed_execution_result=failed_execution_result,
        error_type=error_type,
        error_message=error_message,
        successful_code=successful_code,
        successful_execution_result=successful_execution_result,
        total_attempts=total_attempts,
        tags=tags or []
    )

    return await memory_manager.save_experience(experience)

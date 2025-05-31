# Define NotebookSystem class as an information sharing mechanism
class NotebookSystem:
    def __init__(self):
        # Store the latest responses from each agent
        self.notebook = {}
        # Store task information
        self.tasks = {}
        
    def save_response(self, agent_name, response_content):
        """Save the latest response from an agent to the notebook"""
        self.notebook[agent_name] = response_content
        
    def save_task(self, agent_name, task_content):
        """Save task information to the notebook"""
        self.tasks[agent_name] = task_content
        
    def format_notebook_for_agent(self, exclude_agent=None):
        """Format notebook content, with option to exclude current agent's own content"""
        formatted_content = ""
        # First add task information
        for agent_name, task in self.tasks.items():
            if exclude_agent and agent_name == exclude_agent:
                continue
            formatted_content += f"<{agent_name}Task>\n{task}\n</{agent_name}Task>\n\n"
        # Then add response information
        for agent_name, content in self.notebook.items():
            # Exclude current agent's content
            if exclude_agent and agent_name == exclude_agent:
                continue
            formatted_content += f"<{agent_name}Response>\n{content}\n</{agent_name}Response>\n\n"
        return formatted_content

    def enrich_message(self, message_content, current_agent):
        """Add other agents' responses to the message"""
        notebook_content = self.format_notebook_for_agent(exclude_agent=current_agent)
        if notebook_content:
            enriched_message = f"{message_content}\n\nHere are the latest responses from other agents for your reference:\n{notebook_content}"
            return enriched_message
        return message_content
import re


class ToolSelector:
    
    def __init__(self, client):
        self.client = client
        self.tool_selection_history = []
    
    def select_tool(self, obs, plan, history_messages):
        """
        Ask the master LLM to select which tool/reasoner to use.
        
        Args:
            obs: Current observation
            plan: Current plan (if any)
            history_messages: Conversation history
            
        Returns:
            str: Selected tool name ('reactive_actor', 'cot', or 'heuristic_script')
        """
        instruction = self._get_tool_selection_prompt(obs, plan)
        
        messages = history_messages.copy()
        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + instruction
        
        response = self.client.generate(messages)
        tool_name = self._parse_tool_selection(response.completion)
        
        self.tool_selection_history.append(tool_name)
        
        return tool_name
    
    def _get_tool_selection_prompt(self, obs, plan):
        """
        Generate prompt for tool selection.
        
        Returns:
            str: Tool selection instruction
        """
        return """Based on the current observation and your plan, select which reasoning tool to use:

1. reactive_actor - Fast, direct action selection without explicit reasoning
2. cot - Chain-of-thought reasoning with step-by-step thinking
3. heuristic_script - Rule-based policy using domain-specific heuristics

Consider:
- Task complexity (simple → reactive_actor, complex → cot)
- Need for structured reasoning (yes → cot, no → reactive_actor)
- Availability of good heuristics (yes → heuristic_script)

Output your selection in the following format:
<tool>TOOL_NAME</tool>
Replace TOOL_NAME with one of: reactive_actor, cot, heuristic_script"""
    
    def _parse_tool_selection(self, response_text):
        """
        Extract tool selection from LLM response.
        
        Args:
            response_text: LLM response
            
        Returns:
            str: Tool name
        """
        tool_pattern = r'<tool>(.*?)</tool>'
        tool_match = re.search(tool_pattern, response_text, re.IGNORECASE | re.DOTALL)
        
        if tool_match:
            tool_name = tool_match.group(1).strip().lower()
            
            valid_tools = ['reactive_actor', 'cot', 'heuristic_script']
            if tool_name in valid_tools:
                return tool_name
        
        return 'reactive_actor'
    
    def get_tool_distribution(self):
        """
        Get distribution of tool selections.
        
        Returns:
            dict: Tool selection statistics
        """
        if not self.tool_selection_history:
            return {}
        
        tool_counts = {}
        for tool in self.tool_selection_history:
            tool_counts[tool] = tool_counts.get(tool, 0) + 1
        
        total = len(self.tool_selection_history)
        tool_distribution = {
            tool: count / total 
            for tool, count in tool_counts.items()
        }
        
        return {
            'tool_counts': tool_counts,
            'tool_distribution': tool_distribution,
            'total_selections': total
        }
    
    def reset(self):
        """
        Reset tool selection history.
        """
        self.tool_selection_history = []
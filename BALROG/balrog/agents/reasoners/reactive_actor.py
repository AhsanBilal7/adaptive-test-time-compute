import re


class ReactiveActorReasoner:
    
    def __init__(self, client):
        self.client = client
    
    def generate_action(self, messages, plan=None):
        """
        Generate action using reactive (fast) reasoning.
        
        Args:
            messages: Conversation history
            plan: Current plan (optional)
            
        Returns:
            str: Generated action
        """
        instruction = self._get_reactive_instruction(plan)
        
        messages_copy = messages.copy()
        if messages_copy and messages_copy[-1].role == "user":
            messages_copy[-1].content += "\n\n" + instruction
        
        response = self.client.generate(messages_copy)
        action = self._extract_action(response.completion)
        
        return action
    
    def _get_reactive_instruction(self, plan):
        """
        Get instruction for reactive action generation.
        
        Returns:
            str: Reactive instruction
        """
        base = "Based on your observation"
        if plan:
            base += " and plan"
        
        return f"""{base}, choose exactly ONE action from the allowed actions.

React quickly and directly to the current situation.

Output only the action, nothing else."""
    
    def _extract_action(self, response_text):
        """
        Extract action from response.
        
        Args:
            response_text: LLM response
            
        Returns:
            str: Extracted action
        """
        response_text = re.sub(r'<[^>]+>', '', response_text)
        
        response_text = re.sub(
            r'^[\s]*(?:ACTION|Action)[\s]*[:=]?\s*',
            '',
            response_text,
            flags=re.IGNORECASE
        )
        
        action = response_text.split('\n')[0].strip()
        
        return action
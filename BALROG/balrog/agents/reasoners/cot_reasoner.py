import re


class CoTReasoner:
    
    def __init__(self, client):
        self.client = client
        self.reasoning_history = []
    
    def generate_action(self, messages, plan=None):
        """
        Generate action using chain-of-thought reasoning.
        
        Args:
            messages: Conversation history
            plan: Current plan (optional)
            
        Returns:
            tuple: (action, reasoning)
        """
        instruction = self._get_cot_instruction(plan)
        
        messages_copy = messages.copy()
        if messages_copy and messages_copy[-1].role == "user":
            messages_copy[-1].content += "\n\n" + instruction
        
        response = self.client.generate(messages_copy)
        action, reasoning = self._extract_action_and_reasoning(response.completion)
        
        self.reasoning_history.append(reasoning)
        
        return action, reasoning
    
    def _get_cot_instruction(self, plan):
        """
        Get instruction for chain-of-thought reasoning.
        
        Returns:
            str: CoT instruction
        """
        base = "Look at your observation"
        if plan:
            base += " and plan"
        
        return f"""{base}, then think step-by-step about what to do next.

Output your reasoning in the following format:
<reasoning>
Step 1: [Analyze current situation]
Step 2: [Consider options]
Step 3: [Decide on action]
</reasoning>

After your reasoning, output exactly ONE action from the allowed actions.
Output format:
<reasoning>YOUR STEP-BY-STEP THINKING</reasoning>
ACTION: [your chosen action]"""
    
    def _extract_action_and_reasoning(self, response_text):
        """
        Extract action and reasoning from response.
        
        Args:
            response_text: LLM response
            
        Returns:
            tuple: (action, reasoning)
        """
        reasoning_pattern = r'<reasoning>(.*?)</reasoning>'
        reasoning_match = re.search(reasoning_pattern, response_text, re.IGNORECASE | re.DOTALL)
        
        reasoning = None
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
            text_after_reasoning = response_text[reasoning_match.end():]
            action_text = text_after_reasoning.strip()
        else:
            action_text = response_text.strip()
        
        action = self._extract_action(action_text)
        
        return action, reasoning
    
    def _extract_action(self, action_text):
        """
        Extract action from text.
        
        Args:
            action_text: Text containing action
            
        Returns:
            str: Extracted action
        """
        action_text = re.sub(r'<[^>]+>', '', action_text)
        
        action_text = re.sub(
            r'^[\s]*(?:ACTION|Action)[\s]*[:=]?\s*',
            '',
            action_text,
            flags=re.IGNORECASE
        )
        
        action = action_text.split('\n')[0].strip()
        
        return action
    
    def get_reasoning_history(self):
        """
        Get all reasoning steps from history.
        
        Returns:
            list: List of reasoning steps
        """
        return [r for r in self.reasoning_history if r is not None]
    
    def reset(self):
        """
        Reset reasoning history.
        """
        self.reasoning_history = []
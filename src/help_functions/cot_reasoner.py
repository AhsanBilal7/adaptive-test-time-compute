"""Chain-of-Thought (CoT) reasoner that produces one step-delimited reasoning action."""

import re


class CoTReasoner:
    """Chain-of-Thought reasoner."""
    
    def __init__(self, client):
        self.client = client
        self.reasoning_history = []
    
    def generate_action(self, messages, instruction_prompt):
        """Generate the next reasoning action; returns (action, reasoning)."""
        messages_copy = messages.copy()
        
        if messages_copy:
            last_msg = messages_copy[-1]
            role = last_msg.get("role", "") if isinstance(last_msg, dict) else getattr(last_msg, "role", "")
            
            if role == "user":
                content = last_msg.get("content", "") if isinstance(last_msg, dict) else getattr(last_msg, "content", "")
                new_content = content + "\n\n" + instruction_prompt
                
                if isinstance(last_msg, dict):
                    messages_copy[-1] = {"role": role, "content": new_content}
                else:
                    messages_copy[-1] = {"role": role, "content": new_content}
        
        response = self.client.generate(messages_copy)
        completion = response.completion if hasattr(response, "completion") else str(response)
        
        action, reasoning = self._extract_action_and_reasoning(completion)
        self.reasoning_history.append(reasoning)
        

        return action, reasoning
    
    def _extract_action_and_reasoning(self, text):
        reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.IGNORECASE | re.DOTALL)
        
        reasoning = None
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
            action_text = text[reasoning_match.end():].strip()
        else:
            action_text = text.strip()
        
        action = self._extract_action(action_text)
        return action, reasoning
    

    def _extract_action(self, text):
        action_match = re.search(r"<action>(.*?)</action>", text, re.IGNORECASE | re.DOTALL)
        if action_match:
            return action_match.group(1).strip()
        
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"^[\s]*(?:ACTION|Action)[\s]*[:=]?\s*", "", text, flags=re.IGNORECASE)
        action = text.split("\n")[0].strip()
        
        return action
    
    def get_reasoning_history(self):
        return [r for r in self.reasoning_history if r is not None]
    
    def reset(self):
        self.reasoning_history = []
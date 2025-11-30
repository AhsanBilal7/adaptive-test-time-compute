import re


class ReactiveActorReasoner:
    
    def __init__(self, client):
        self.client = client
    
    def generate_action(self, messages, instruction_prompt):
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
        return action, reasoning
    
    def _extract_action_and_reasoning(self, text):
        reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', text, re.IGNORECASE | re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else None
        
        action_match = re.search(r'<action>(.*?)</action>', text, re.IGNORECASE | re.DOTALL)
        
        if action_match:
            action = action_match.group(1).strip()
        else:
            if reasoning_match:
                action_text = text[reasoning_match.end():].strip()
            else:
                action_text = text.strip()
            
            action = self._extract_action(action_text)
        
        return action, reasoning
    
    def _extract_action(self, text):
        answer_match = re.search(r'<answer>(.*?)</answer>', text, re.IGNORECASE | re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()
        
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'^[\s]*(?:ACTION|Action)[\s]*[:=]?\s*', '', text, flags=re.IGNORECASE)
        action = text.split('\n')[0].strip()
        
        return action
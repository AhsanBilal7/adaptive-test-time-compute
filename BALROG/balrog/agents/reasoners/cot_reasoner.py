from email.mime import base
import re


class CoTReasoner:
    
    def __init__(self, client):
        self.client = client
        self.reasoning_history = []
    
    def _get_message_role(self, msg):
        """
        Get role from message (handles both dict and object formats).
        
        Args:
            msg: Message (dict or object)
            
        Returns:
            str: Role of the message
        """
        if isinstance(msg, dict):
            return msg.get("role", "")
        else:
            return getattr(msg, "role", "")
    
    def _get_message_content(self, msg):
        """
        Get content from message (handles both dict and object formats).
        
        Args:
            msg: Message (dict or object)
            
        Returns:
            str: Content of the message
        """
        if isinstance(msg, dict):
            return msg.get("content", "")
        else:
            return getattr(msg, "content", "")
    
    def _set_message_content(self, msg, content):
        """
        Set content for message (handles both dict and object formats).
        
        Args:
            msg: Message (dict or object)
            content: New content
            
        Returns:
            Modified message (dict format)
        """
        if isinstance(msg, dict):
            msg["content"] = content
            return msg
        else:
            # Convert object to dict format
            return {
                "role": getattr(msg, "role", "user"),
                "content": content
            }
    
    def generate_action(self, messages, plan=None):
        """
        Generate action using chain-of-thought reasoning.
        
        Args:
            messages: Conversation history (list of dicts or objects)
            plan: Current plan (optional)
            
        Returns:
            tuple: (action, reasoning)
        """
        instruction = self._get_cot_instruction(plan)
        
        messages_copy = messages.copy()
        # print("messages_copy before adding instruction:", messages_copy)  # Debugging line
        
        # Add instruction to last user message
        if messages_copy:
            last_msg = messages_copy[-1]
            last_role = self._get_message_role(last_msg)
            
            if last_role == "user":
                current_content = self._get_message_content(last_msg)
                new_content = current_content + "\n\n" + instruction
                messages_copy[-1] = self._set_message_content(last_msg, new_content)
        
        response = self.client.generate(messages_copy)
        action, reasoning = self._extract_action_and_reasoning(response.completion)
        
        print("messages_copy after adding instruction:", messages_copy)  # Debugging line
        print("CoT Reasoner response:", response.completion)  # Debugging line
        print("CoT Reasoner action:", action)  # Debugging line
        print("CoT Reasoner reasoning:", reasoning)  # Debugging line

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
        
        return f"""{base}, then decide the best next move based on the current environment, inventory, and visible entities.

        After your reasoning, output exactly ONE action from the allowed actions below.
        Choose exactly one action from:
        Noop, Move West, Move East, Move North, Move South,
        Do, Sleep,
        Place Stone, Place Table, Place Furnace, Place Plant,
        Make Wood Pickaxe, Make Stone Pickaxe, Make Iron Pickaxe,
        Make Wood Sword, Make Stone Sword, Make Iron Sword.

        Output format:
        <reasoning>
        [Your brief step-by-step thought process]
        </reasoning>
        ACTION: <action_name>

        Return only the reasoning and the ACTION line, nothing else."""
    
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
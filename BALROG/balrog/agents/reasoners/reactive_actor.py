import re

class ReactiveActorReasoner:
    
    def __init__(self, client):
        self.client = client
    
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
        Generate action using reactive (fast) reasoning.
        
        Args:
            messages: Conversation history (list of dicts or objects)
            plan: Current plan (optional)
            
        Returns:
            str: Generated action
        """
        instruction = self._get_reactive_instruction(plan)
        
        messages_copy = messages.copy()
        
        # Add instruction to last user message - handles both dict and object formats
        if messages_copy:
            last_msg = messages_copy[-1]
            last_role = self._get_message_role(last_msg)
            
            if last_role == "user":
                current_content = self._get_message_content(last_msg)
                new_content = current_content + "\n\n" + instruction
                messages_copy[-1] = self._set_message_content(last_msg, new_content)
        
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

                Respond quickly and instinctively to the current environment based on what you see.

                Output format:
                ACTION: [Noop|Move West|Move East|Move North|Move South|Do|Sleep|Place Stone|Place Table|Place Furnace|Place Plant|Make Wood Pickaxe|Make Stone Pickaxe|Make Iron Pickaxe|Make Wood Sword|Make Stone Sword|Make Iron Sword]

                Output only the action line, nothing else."""

    
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
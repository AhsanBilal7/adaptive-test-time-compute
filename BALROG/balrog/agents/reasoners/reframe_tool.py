"""Tool that reformulates questions or plans for clarity."""

class ReframeTool:
    """Tool to reformulate questions or plans for better clarity."""
    
    def __init__(self, client):
        self.client = client
        self.reframe_history = []
    
    def reframe(self, question, context):
        """
        Reframe the question to highlight key reasoning challenges.
        
        Args:
            question: Original question or task description
            context: Additional context (plan, observations, etc.)
            
        Returns:
            str: Reframed question
        """
        prompt = [
            {
                "role": "system", 
                "content": "Reframe the question to highlight the key reasoning challenge and clarify ambiguities."
            },
            {
                "role": "user", 
                "content": f"Question: {question}\nContext: {str(context)[:800]}"
            }
        ]
        
        try:
            resp = self.client.generate(prompt)
            reframed = resp.completion if hasattr(resp, "completion") else str(resp)
            reframed = reframed.strip()
        except Exception:
            # Fallback: return original
            reframed = question
        
        self.reframe_history.append({
            "original": question,
            "reframed": reframed
        })
        
        return reframed
    
    def reset(self):
        """Reset reframe history."""
        self.reframe_history = []
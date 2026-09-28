"""PRM-based numeric verification tool."""

class NumericVerifier:
    """PRM-based numeric verification tool."""
    
    def __init__(self, prm_model):
        self.prm_model = prm_model
        self.verification_history = []
    
    def verify(self, messages, obs, plan, action):
        """
        Scores the numeric correctness of the action output.
        
        Args:
            messages: Conversation history
            obs: Current observation
            plan: Current plan
            action: Action to verify
            
        Returns:
            dict: Verification result with verdict, score, and comment
        """
        try:
            score = self.prm_model.score_response(action, obs, plan, context="Numeric verification")
            verdict = "correct" if score > 0.8 else "incorrect"
        except Exception:
            score, verdict = 0.0, "error"
        
        result = {
            "verdict": verdict,
            "score": score,
            "comment": f"Numeric verification score={score:.2f}"
        }
        
        self.verification_history.append(result)
        return result
    
    def reset(self):
        """Reset verification history."""
        self.verification_history = []
class VerifierTool:
    """General-purpose PRM-based verification tool for textual correctness."""
    
    def __init__(self, prm_model):
        self.prm_model = prm_model
        self.verification_history = []
    
    def verify(self, messages, obs, plan, text):
        """
        Verify the correctness of the given text using PRM.
        
        Args:
            messages: Conversation history
            obs: Current observation
            plan: Current plan
            text: Text to verify (reasoning, answer, etc.)
            
        Returns:
            dict: Verification result with verdict, score, and comment
        """
        try:
            score = self.prm_model.score_response(text, obs, plan, context="General verification")
            
            # Thresholds for verdict
            if score > 0.85:
                verdict = "highly_confident"
            elif score > 0.7:
                verdict = "confident"
            elif score > 0.5:
                verdict = "uncertain"
            else:
                verdict = "likely_incorrect"
                
        except Exception:
            score, verdict = 0.0, "error"
        
        result = {
            "verdict": verdict,
            "score": score,
            "comment": f"Verification: {verdict} (score={score:.2f})"
        }
        
        self.verification_history.append(result)
        return result
    
    def reset(self):
        """Reset verification history."""
        self.verification_history = []
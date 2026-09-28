"""Tool that compresses long reasoning chains."""

class SummarizerTool:
    """Tool to compress long reasoning chains into concise summaries."""
    
    def __init__(self, client):
        self.client = client
        self.summary_history = []
    
    def summarize(self, text):
        """
        Summarize the given text using the LLM.
        
        Args:
            text: Text to summarize (reasoning chain, context, etc.)
            
        Returns:
            str: Concise summary
        """
        prompt = [
            {"role": "system", "content": "Summarize the following reasoning concisely, preserving key insights."},
            {"role": "user", "content": str(text)[:2000]}  # Truncate to prevent overload
        ]
        
        try:
            resp = self.client.generate(prompt)
            summary = resp.completion if hasattr(resp, "completion") else str(resp)
            summary = summary.strip()
        except Exception:
            # Fallback: simple truncation
            summary = str(text)[:256] + "..."
        
        self.summary_history.append({
            "original_length": len(str(text)),
            "summary_length": len(summary),
            "summary": summary
        })
        
        return summary
    
    def reset(self):
        """Reset summary history."""
        self.summary_history = []
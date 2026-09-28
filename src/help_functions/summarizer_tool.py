"""Summarizer tool that compresses long reasoning trajectories."""

class SummarizerTool:
    """Compresses long reasoning with the LLM."""
    
    def __init__(self, client):
        self.client = client
        self.summary_history = []
    
    def summarize(self, text):
        prompt = [
            {"role": "system", "content": "Summarize the following reasoning concisely, preserving key insights."},
            {"role": "user", "content": str(text)}
        ]
        
        resp = self.client.generate(prompt)
        summary = resp.completion if hasattr(resp, "completion") else str(resp)
        summary = summary.strip()
        
        self.summary_history.append({
            "original_length": len(str(text)),
            "summary_length": len(summary),
            "summary": summary
        })
        
        return summary
    
    def reset(self):
        self.summary_history = []

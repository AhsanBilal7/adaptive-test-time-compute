class NumericVerifier:
    
    def __init__(self, prm_model):
        self.prm_model = prm_model
        self.verification_history = []
    
    def verify(self, messages, obs, plan, action, prm_scoring_prompt):
        score = self.prm_model.score_response(action, obs, plan, "Numeric verification", prm_scoring_prompt)
        verdict = "correct" if score > 0.8 else "incorrect"
        
        result = {
            "verdict": verdict,
            "score": score,
            "comment": f"Numeric verification score={score:.2f}"
        }
        
        self.verification_history.append(result)
        return result
    
    def reset(self):
        self.verification_history = []


class VerifierTool:
    
    def __init__(self, prm_model):
        self.prm_model = prm_model
        self.verification_history = []
    
    def verify(self, messages, obs, plan, text, prm_scoring_prompt):
        score = self.prm_model.score_response(text, obs, plan, "General verification", prm_scoring_prompt)
        
        if score > 0.85:
            verdict = "highly_confident"
        elif score > 0.7:
            verdict = "confident"
        elif score > 0.5:
            verdict = "uncertain"
        else:
            verdict = "likely_incorrect"
        
        result = {
            "verdict": verdict,
            "score": score,
            "comment": f"Verification: {verdict} (score={score:.2f})"
        }
        
        self.verification_history.append(result)
        return result
    
    def reset(self):
        self.verification_history = []


class SummarizerTool:
    
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


class ReframeTool:
    
    def __init__(self, client):
        self.client = client
        self.reframe_history = []
    
    def reframe(self, question, context):
        prompt = [
            {
                "role": "system",
                "content": "Reframe the question to highlight the key reasoning challenge and clarify ambiguities."
            },
            {
                "role": "user",
                "content": f"Question: {question}\nContext: {str(context)}"
            }
        ]
        
        resp = self.client.generate(prompt)
        reframed = resp.completion if hasattr(resp, "completion") else str(resp)
        reframed = reframed.strip()
        
        self.reframe_history.append({
            "original": question,
            "reframed": reframed
        })
        
        return reframed
    
    def reset(self):
        self.reframe_history = []


class WebSearchTool:
    
    def __init__(self, client=None):
        self.client = client
        self.search_history = []
    
    def search(self, query):
        result = f"[Web search placeholder for query: {str(query)[:100]}]"
        
        self.search_history.append({
            "query": str(query),
            "result": result
        })
        
        return result
    
    def reset(self):
        self.search_history = []
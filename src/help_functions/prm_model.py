"""LLM-prompted step scorer used inside compute strategies and verification tools."""

import re


class PRMModel:
    """Scores a reasoning step or response in [0, 1] by prompting the LLM."""
    
    def __init__(self, client):
        self.client = client
        self.scoring_history = []
    
    def score_response(self, response_text, obs, plan, context, prm_scoring_prompt):
        prompt = self._build_scoring_prompt(response_text, obs, plan, context, prm_scoring_prompt)
        
        messages = [{"role": "user", "content": prompt}]
        response = self.client.generate(messages)
        text = response.completion if hasattr(response, "completion") else str(response)
        score = self._parse_score(text.strip())
        
        self.scoring_history.append({'response': response_text, 'score': score})
        return score
    
    def score_multiple(self, responses, obs, plan, context, prm_scoring_prompt):
        scores = []
        for response_text in responses:
            score = self.score_response(response_text, obs, plan, context, prm_scoring_prompt)
            scores.append(score)
        return scores
    
    def select_best(self, responses, obs, plan, context, prm_scoring_prompt):
        scores = self.score_multiple(responses, obs, plan, context, prm_scoring_prompt)
        best_idx = scores.index(max(scores))
        return responses[best_idx], scores[best_idx], scores
    
    def _build_scoring_prompt(self, response_text, obs, plan, context, prm_prompt):
        parts = []
        
        if plan:
            parts.append(f"Current Plan:\n{plan}")
        
        parts.append(f"Given Problem:\n{obs}")
        
        if context:
            parts.append(f"Context:\n{context}")
        
        parts.append(f"Response to Evaluate:\n{response_text}")
        parts.append(prm_prompt)
        
        return "\n\n".join(parts)
    
    def _parse_score(self, text):
        score_match = re.search(r'<score>(.*?)</score>', text, re.IGNORECASE | re.DOTALL)
        
        if score_match:
            score = float(score_match.group(1).strip())
            return max(0.0, min(1.0, score))
        
        numeric_match = re.search(r'\b([0-1]?\.\d+)\b', text)
        if numeric_match:
            score = float(numeric_match.group(1))
            return max(0.0, min(1.0, score))
        
        return 0.5
    
    def get_scoring_stats(self):
        if not self.scoring_history:
            return {}
        
        scores = [entry['score'] for entry in self.scoring_history]
        
        return {
            'total_scored': len(scores),
            'avg_score': sum(scores) / len(scores),
            'min_score': min(scores),
            'max_score': max(scores)
        }
    
    def reset(self):
        self.scoring_history = []
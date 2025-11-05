class PRMModel:
    
    def __init__(self, client):
        self.client = client
        self.scoring_history = []
    
    def score_response(self, response_text, obs, plan, context=""):
        """
        Score a response using the PRM model with robust error handling.
        
        Args:
            response_text: Response to score
            obs: Current observation
            plan: Current plan
            context: Additional context
            
        Returns:
            float: Score between 0 and 1
        """
        prompt = self._build_scoring_prompt(response_text, obs, plan, context)
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.client.generate(messages)
            text = response.completion if hasattr(response, "completion") else str(response)
            score = self._parse_score(text.strip())
        except Exception as e:
            print(f"[PRMModel] Error during scoring: {e}")
            score = 0.0
        
        self.scoring_history.append({
            'response': response_text,
            'score': score
        })
        
        return score
    
    def score_multiple(self, responses, obs, plan, context=""):
        """
        Score multiple responses and return scores.
        
        Args:
            responses: List of response texts
            obs: Current observation
            plan: Current plan
            context: Additional context
            
        Returns:
            list: List of scores
        """
        scores = []
        for response_text in responses:
            score = self.score_response(response_text, obs, plan, context)
            scores.append(score)
        
        return scores
    
    def select_best(self, responses, obs, plan, context=""):
        """
        Score multiple responses and return the best one.
        
        Args:
            responses: List of response texts
            obs: Current observation
            plan: Current plan
            context: Additional context
            
        Returns:
            tuple: (best_response, best_score, all_scores)
        """
        scores = self.score_multiple(responses, obs, plan, context)
        
        best_idx = scores.index(max(scores))
        best_response = responses[best_idx]
        best_score = scores[best_idx]
        
        return best_response, best_score, scores
    
    def _build_scoring_prompt(self, response_text, obs, plan, context):
        """
        Build prompt for scoring a response.
        
        Returns:
            str: Scoring prompt
        """
        prompt_parts = []
        
        if plan:
            prompt_parts.append(f"Current Plan:\n{plan}")
        
        prompt_parts.append(f"Observation:\n{obs}")
        
        if context:
            prompt_parts.append(f"Context:\n{context}")
        
        prompt_parts.append(f"Response to Evaluate:\n{response_text}")
        
        prompt_parts.append("""
Evaluate this response based on:
1. Alignment with the plan and goals
2. Appropriateness for the current observation
3. Likelihood of making progress toward task completion
4. Correctness and safety of the proposed action

Provide a score between 0.0 and 1.0:
- 0.0-0.3: Poor response (unhelpful, incorrect, or counterproductive)
- 0.3-0.5: Below average (some issues, suboptimal)
- 0.5-0.7: Average (reasonable, acceptable)
- 0.7-0.9: Good (appropriate, effective)
- 0.9-1.0: Excellent (optimal, highly aligned)

Output your score in the format:
<score>X.XX</score>
Replace X.XX with your numeric score.""")
        
        return "\n\n".join(prompt_parts)
    
    def _parse_score(self, response_text):
        """
        Extract score from LLM response.
        
        Args:
            response_text: LLM response
            
        Returns:
            float: Score between 0 and 1
        """
        import re
        
        score_pattern = r'<score>(.*?)</score>'
        score_match = re.search(score_pattern, response_text, re.IGNORECASE | re.DOTALL)
        
        if score_match:
            try:
                score = float(score_match.group(1).strip())
                return max(0.0, min(1.0, score))
            except ValueError:
                pass
        
        numeric_pattern = r'\b([0-1]?\.\d+)\b'
        numeric_match = re.search(numeric_pattern, response_text)
        
        if numeric_match:
            try:
                score = float(numeric_match.group(1))
                return max(0.0, min(1.0, score))
            except ValueError:
                pass
        
        return 0.5
    
    def get_scoring_stats(self):
        """
        Get statistics about scoring history.
        
        Returns:
            dict: Scoring statistics
        """
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
        """
        Reset scoring history.
        """
        self.scoring_history = []
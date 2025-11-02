import re


class ComputeSelector:
    
    def __init__(self, client):
        self.client = client
        self.compute_selection_history = []
    
    def select_compute_strategy(self, obs, plan, tool_name, history_messages):
        """
        Ask the master LLM to select which compute strategy to use.
        
        Args:
            obs: Current observation
            plan: Current plan (if any)
            tool_name: Selected tool/reasoner
            history_messages: Conversation history
            
        Returns:
            dict: Compute strategy config with 'strategy' and 'params'
        """
        instruction = self._get_compute_selection_prompt(tool_name)
        
        messages = history_messages.copy()
        if messages and messages[-1].role == "user":
            messages[-1].content += "\n\n" + instruction
        
        response = self.client.generate(messages)
        compute_config = self._parse_compute_selection(response.completion)
        
        self.compute_selection_history.append(compute_config['strategy'])
        
        return compute_config
    
    def _get_compute_selection_prompt(self, tool_name):
        """
        Generate prompt for compute strategy selection.
        
        Returns:
            str: Compute strategy selection instruction
        """
        return f"""You have selected the '{tool_name}' reasoning tool. Now select a compute strategy:

1. best_of_n - Generate N responses and select the best using PRM scoring
   - Good for high-quality single outputs
   - Moderate computational cost
   - Recommended N: 3-5

2. beam_search - Maintain top-K candidates at each step using PRM scoring
   - Good for multi-step reasoning
   - Higher computational cost
   - Recommended beam_width: 3-5

3. lookahead - Simulate future steps and score with PRM
   - Good for planning-heavy tasks
   - Highest computational cost
   - Recommended depth: 2-3

Consider:
- Task difficulty (harder → more compute)
- Available resources (limited → best_of_n with small N)
- Multi-step reasoning needs (yes → beam_search or lookahead)

Output your selection in the following format:
<compute>STRATEGY:PARAM_VALUE</compute>
Examples:
- <compute>best_of_n:5</compute>
- <compute>beam_search:3</compute>
- <compute>lookahead:2</compute>"""
    
    def _parse_compute_selection(self, response_text):
        """
        Extract compute strategy from LLM response.
        
        Args:
            response_text: LLM response
            
        Returns:
            dict: Compute configuration
        """
        compute_pattern = r'<compute>(.*?)</compute>'
        compute_match = re.search(compute_pattern, response_text, re.IGNORECASE | re.DOTALL)
        
        if compute_match:
            compute_str = compute_match.group(1).strip().lower()
            
            if ':' in compute_str:
                strategy, param = compute_str.split(':', 1)
                strategy = strategy.strip()
                try:
                    param_value = int(param.strip())
                except ValueError:
                    param_value = 3
                
                valid_strategies = ['best_of_n', 'beam_search', 'lookahead']
                if strategy in valid_strategies:
                    return {
                        'strategy': strategy,
                        'param': param_value
                    }
        
        return {
            'strategy': 'best_of_n',
            'param': 3
        }
    
    def get_compute_distribution(self):
        """
        Get distribution of compute strategy selections.
        
        Returns:
            dict: Compute strategy statistics
        """
        if not self.compute_selection_history:
            return {}
        
        strategy_counts = {}
        for strategy in self.compute_selection_history:
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        total = len(self.compute_selection_history)
        strategy_distribution = {
            strategy: count / total 
            for strategy, count in strategy_counts.items()
        }
        
        return {
            'strategy_counts': strategy_counts,
            'strategy_distribution': strategy_distribution,
            'total_selections': total
        }
    
    def reset(self):
        """
        Reset compute selection history.
        """
        self.compute_selection_history = []
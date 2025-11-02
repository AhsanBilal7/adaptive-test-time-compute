class ComputeStrategy:
    
    def __init__(self, prm_model):
        self.prm_model = prm_model
    
    def execute(self, reasoner, messages, obs, plan, config):
        """
        Execute compute strategy.
        
        Args:
            reasoner: Reasoner to use
            messages: Conversation history
            obs: Current observation
            plan: Current plan
            config: Strategy configuration
            
        Returns:
            tuple: (action, metadata)
        """
        raise NotImplementedError


class BestOfNStrategy(ComputeStrategy):
    
    def execute(self, reasoner, messages, obs, plan, config):
        """
        Execute Best-of-N strategy: generate N responses and select best.
        
        Args:
            reasoner: Reasoner to use
            messages: Conversation history
            obs: Current observation
            plan: Current plan
            config: Strategy configuration with 'param' (N value)
            
        Returns:
            tuple: (best_action, metadata)
        """
        n = config['param']
        
        responses = []
        for i in range(n):
            if hasattr(reasoner, 'generate_action'):
                if reasoner.__class__.__name__ == 'HeuristicScriptReasoner':
                    action, rule = reasoner.generate_action(obs, plan)
                    responses.append(action)
                elif reasoner.__class__.__name__ == 'CoTReasoner':
                    action, reasoning = reasoner.generate_action(messages, plan)
                    responses.append(action)
                else:
                    action = reasoner.generate_action(messages, plan)
                    responses.append(action)
        
        best_response, best_score, all_scores = self.prm_model.select_best(
            responses, obs, plan, context="Best-of-N selection"
        )
        
        metadata = {
            'strategy': 'best_of_n',
            'n': n,
            'all_responses': responses,
            'all_scores': all_scores,
            'best_score': best_score
        }
        
        return best_response, metadata


class BeamSearchStrategy(ComputeStrategy):
    
    def execute(self, reasoner, messages, obs, plan, config):
        """
        Execute Beam Search strategy: maintain top-K candidates.
        
        Args:
            reasoner: Reasoner to use
            messages: Conversation history
            obs: Current observation
            plan: Current plan
            config: Strategy configuration with 'param' (beam width)
            
        Returns:
            tuple: (best_action, metadata)
        """
        beam_width = config['param']
        
        candidates = []
        for i in range(beam_width * 2):
            if hasattr(reasoner, 'generate_action'):
                if reasoner.__class__.__name__ == 'HeuristicScriptReasoner':
                    action, rule = reasoner.generate_action(obs, plan)
                    candidates.append(action)
                elif reasoner.__class__.__name__ == 'CoTReasoner':
                    action, reasoning = reasoner.generate_action(messages, plan)
                    candidates.append(action)
                else:
                    action = reasoner.generate_action(messages, plan)
                    candidates.append(action)
        
        scores = self.prm_model.score_multiple(
            candidates, obs, plan, context="Beam search"
        )
        
        scored_candidates = list(zip(candidates, scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        top_k = scored_candidates[:beam_width]
        best_action = top_k[0][0]
        best_score = top_k[0][1]
        
        metadata = {
            'strategy': 'beam_search',
            'beam_width': beam_width,
            'top_k_candidates': [c[0] for c in top_k],
            'top_k_scores': [c[1] for c in top_k],
            'best_score': best_score
        }
        
        return best_action, metadata


class LookaheadStrategy(ComputeStrategy):
    
    def execute(self, reasoner, messages, obs, plan, config):
        """
        Execute Look-ahead strategy: simulate future steps and score.
        
        Args:
            reasoner: Reasoner to use
            messages: Conversation history
            obs: Current observation
            plan: Current plan
            config: Strategy configuration with 'param' (depth)
            
        Returns:
            tuple: (best_action, metadata)
        """
        depth = config['param']
        
        initial_candidates = []
        for i in range(3):
            if hasattr(reasoner, 'generate_action'):
                if reasoner.__class__.__name__ == 'HeuristicScriptReasoner':
                    action, rule = reasoner.generate_action(obs, plan)
                    initial_candidates.append(action)
                elif reasoner.__class__.__name__ == 'CoTReasoner':
                    action, reasoning = reasoner.generate_action(messages, plan)
                    initial_candidates.append(action)
                else:
                    action = reasoner.generate_action(messages, plan)
                    initial_candidates.append(action)
        
        best_action = None
        best_cumulative_score = -1
        lookahead_results = []
        
        for candidate in initial_candidates:
            cumulative_score = 0
            trajectory = [candidate]
            
            current_obs = obs
            for step in range(depth):
                step_score = self.prm_model.score_response(
                    candidate, current_obs, plan, 
                    context=f"Lookahead step {step+1}"
                )
                cumulative_score += step_score
                
                if step < depth - 1:
                    if hasattr(reasoner, 'generate_action'):
                        if reasoner.__class__.__name__ == 'HeuristicScriptReasoner':
                            next_action, _ = reasoner.generate_action(current_obs, plan)
                        elif reasoner.__class__.__name__ == 'CoTReasoner':
                            next_action, _ = reasoner.generate_action(messages, plan)
                        else:
                            next_action = reasoner.generate_action(messages, plan)
                        trajectory.append(next_action)
            
            avg_score = cumulative_score / depth
            lookahead_results.append({
                'action': candidate,
                'trajectory': trajectory,
                'cumulative_score': cumulative_score,
                'avg_score': avg_score
            })
            
            if avg_score > best_cumulative_score:
                best_cumulative_score = avg_score
                best_action = candidate
        
        metadata = {
            'strategy': 'lookahead',
            'depth': depth,
            'lookahead_results': lookahead_results,
            'best_score': best_cumulative_score
        }
        
        return best_action, metadata


def get_compute_strategy(strategy_name, prm_model):
    """
    Factory function to get compute strategy.
    
    Args:
        strategy_name: Name of strategy ('best_of_n', 'beam_search', 'lookahead')
        prm_model: PRM model instance
        
    Returns:
        ComputeStrategy: Strategy instance
    """
    strategies = {
        'best_of_n': BestOfNStrategy,
        'beam_search': BeamSearchStrategy,
        'lookahead': LookaheadStrategy
    }
    
    strategy_class = strategies.get(strategy_name, BestOfNStrategy)
    return strategy_class(prm_model)
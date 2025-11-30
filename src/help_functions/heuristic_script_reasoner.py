class HeuristicScriptReasoner:
    
    def __init__(self, policy):
        self.policy = policy
        self.rule_history = []
    
    def generate_action(self, obs, plan=None):
        action, rule_used = self.policy.get_action(obs, plan)
        
        self.rule_history.append({
            "rule": rule_used,
            "action": action
        })
        
        return action, rule_used
    
    def get_rule_distribution(self):
        if not self.rule_history:
            return {}
        
        rule_counts = {}
        for entry in self.rule_history:
            rule = entry["rule"]
            rule_counts[rule] = rule_counts.get(rule, 0) + 1
        
        total = len(self.rule_history)
        rule_distribution = {rule: count / total for rule, count in rule_counts.items()}
        
        return {
            "rule_counts": rule_counts,
            "rule_distribution": rule_distribution,
            "total_actions": total
        }
    
    def reset(self):
        self.rule_history = []
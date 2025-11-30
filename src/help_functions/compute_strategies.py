import random
import inspect


class ComputeStrategy:
    
    def __init__(self, prm_model, rng=None):
        self.prm_model = prm_model
        self.rng = rng if rng is not None else random.Random(0)
    
    def _call_reasoner(self, reasoner, messages, obs, plan, instruction_prompt):
        sig = inspect.signature(reasoner.generate_action)
        params = list(sig.parameters.keys())
        
        if len(params) >= 2:
            first_param = params[0]
            if "message" in first_param or "history" in first_param or "msgs" in first_param:
                out = reasoner.generate_action(messages, instruction_prompt)
            else:
                out = reasoner.generate_action(obs, plan)
        else:
            out = reasoner.generate_action(messages, instruction_prompt)
        
        action = out[0] if isinstance(out, tuple) and len(out) >= 1 else out
        return str(action).strip()
    
    def _score(self, action, messages, obs, plan, prm_scoring_prompt):
        score = self.prm_model.score_response(action, obs, plan, "", prm_scoring_prompt)
        s = float(score)
        if s != s:
            s = 0.0
        return max(0.0, min(1.0, s))
    
    def _dedupe_preserve(self, actions):
        seen = set()
        out = []
        for a in actions:
            if a not in seen:
                seen.add(a)
                out.append(a)
        return out
    
    def execute(self, reasoner, messages, obs, plan, config, instruction_prompt, prm_scoring_prompt):
        raise NotImplementedError


class BestOfNStrategy(ComputeStrategy):
    
    def execute(self, reasoner, messages, obs, plan, config, instruction_prompt, prm_scoring_prompt):
        n = int(config.get("param", 1))
        n = max(1, min(1024, n))
        
        raw_candidates = [
            self._call_reasoner(reasoner, messages, obs, plan, instruction_prompt) 
            for _ in range(n)
        ]
        candidates = self._dedupe_preserve(raw_candidates)
        
        if not candidates:
            candidates = [self._call_reasoner(reasoner, messages, obs, plan, instruction_prompt)]
        
        scores = [self._score(a, messages, obs, plan, prm_scoring_prompt) for a in candidates]
        
        order = list(range(len(candidates)))
        self.rng.shuffle(order)
        best_idx = max(order, key=lambda i: scores[i])
        best_action = candidates[best_idx]
        
        metadata = {
            "strategy": "best_of_n",
            "compute_config": dict(config),
            "num_reasoner_calls": len(raw_candidates) if raw_candidates else 1,
            "num_scored": len(candidates),
            "candidates": candidates,
            "scores": scores,
            "chosen_index": best_idx,
            "deduped": len(candidates) != len(raw_candidates),
        }
        return best_action, metadata


class BeamSearchStrategy(ComputeStrategy):
    
    def execute(self, reasoner, messages, obs, plan, config, instruction_prompt, prm_scoring_prompt):
        beam_width = int(config.get("param", 2))
        beam_width = max(2, min(64, beam_width))
        rounds = int(config.get("rounds", 2))
        rounds = max(1, min(8, rounds))
        
        total_calls = 0
        
        seed = [
            self._call_reasoner(reasoner, messages, obs, plan, instruction_prompt) 
            for _ in range(beam_width * 2)
        ]
        total_calls += len(seed)
        seed = self._dedupe_preserve(seed)
        seed_scores = [self._score(a, messages, obs, plan, prm_scoring_prompt) for a in seed]
        idx = sorted(range(len(seed)), key=lambda i: seed_scores[i], reverse=True)[:beam_width]
        beam = [seed[i] for i in idx]
        beam_scores = [seed_scores[i] for i in idx]
        
        for _ in range(rounds - 1):
            expanded = [
                self._call_reasoner(reasoner, messages, obs, plan, instruction_prompt) 
                for _a in beam
            ]
            total_calls += len(expanded)
            pool = self._dedupe_preserve(beam + expanded)
            pool_scores = [self._score(a, messages, obs, plan, prm_scoring_prompt) for a in pool]
            idx = sorted(range(len(pool)), key=lambda i: pool_scores[i], reverse=True)[:beam_width]
            beam = [pool[i] for i in idx]
            beam_scores = [pool_scores[i] for i in idx]
        
        best_idx = max(range(len(beam)), key=lambda i: beam_scores[i])
        best_action = beam[best_idx]
        
        metadata = {
            "strategy": "beam_search",
            "compute_config": dict(config),
            "num_reasoner_calls": total_calls,
            "beam_width": beam_width,
            "rounds": rounds,
            "beam": beam,
            "beam_scores": beam_scores,
            "chosen_index": best_idx,
        }
        return best_action, metadata


class LookaheadStrategy(ComputeStrategy):
    
    def execute(self, reasoner, messages, obs, plan, config, instruction_prompt, prm_scoring_prompt):
        k = int(config.get("param", 1))
        k = max(1, min(4, k))
        seeds = int(config.get("seeds", 4))
        seeds = max(2, min(32, seeds))
        
        total_calls = 0
        
        seed_actions = [
            self._call_reasoner(reasoner, messages, obs, plan, instruction_prompt) 
            for _ in range(seeds)
        ]
        total_calls += len(seed_actions)
        seed_actions = self._dedupe_preserve(seed_actions)
        
        per_seed_best = []
        all_rollouts = []
        
        for a in seed_actions:
            rollouts = [
                self._call_reasoner(reasoner, messages, obs, plan, instruction_prompt) 
                for _ in range(k)
            ]
            total_calls += len(rollouts)
            rollouts = self._dedupe_preserve(rollouts)
            
            if rollouts:
                scores = [self._score(r, messages, obs, plan, prm_scoring_prompt) for r in rollouts]
            else:
                scores = [self._score(a, messages, obs, plan, prm_scoring_prompt)]
                rollouts = [a]
            
            local_idx = max(range(len(rollouts)), key=lambda i: scores[i])
            per_seed_best.append((a, rollouts[local_idx], scores[local_idx]))
            all_rollouts.append({
                "seed": a, 
                "rollouts": rollouts, 
                "scores": scores, 
                "chosen": local_idx
            })
        
        global_idx = max(range(len(per_seed_best)), key=lambda i: per_seed_best[i][2])
        chosen_seed, chosen_rollout, chosen_score = per_seed_best[global_idx]
        best_action = chosen_rollout
        
        metadata = {
            "strategy": "lookahead",
            "compute_config": dict(config),
            "num_reasoner_calls": total_calls,
            "k": k,
            "seeds": len(seed_actions),
            "seed_actions": seed_actions,
            "per_seed_best": per_seed_best,
            "all_rollouts": all_rollouts,
            "chosen_seed_index": global_idx,
            "chosen_score": chosen_score,
        }
        return best_action, metadata


def get_compute_strategy(strategy_name, prm_model, rng=None):
    strategies = {
        "best_of_n": BestOfNStrategy,
        "beam_search": BeamSearchStrategy,
        "lookahead": LookaheadStrategy,
    }
    cls = strategies.get(strategy_name, BestOfNStrategy)
    return cls(prm_model, rng=rng)
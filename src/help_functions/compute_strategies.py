"""Intra-iteration compute strategies guided by step-level scores: Best-of-N, beam search, and lookahead."""

import random
import inspect


class ComputeStrategy:
    """Base class for compute strategies that rank candidates with the step scorer."""
    
    def __init__(self, prm_model, rng=None):
        self.prm_model = prm_model
        self.rng = rng if rng is not None else random.Random(0)
    
    def _call_reasoner(self, reasoner, messages, obs, plan, instruction_prompt):
        """
        Call the reasoner and extract both action and reasoning.
        Returns: (action, reasoning)
        """
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
        
        # Extract action and reasoning from tuple output
        if isinstance(out, tuple):
            if len(out) >= 2:
                action = str(out[0]).strip()
                reasoning = str(out[1]).strip() if out[1] is not None else ""
            else:
                action = str(out[0]).strip()
                reasoning = ""
        else:
            action = str(out).strip()
            reasoning = ""
        
        return action, reasoning
    
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
        """
        Best-of-N strategy: generate N candidates and select the one with highest score.
        Returns: (best_action, metadata) where metadata includes reasoning texts
        """
        n = int(config.get("param", 1))
        n = max(1, min(1024, n))
        
        raw_candidates_with_reasoning = [
            self._call_reasoner(reasoner, messages, obs, plan, instruction_prompt) 
            for _ in range(n)
        ]
        
        # Separate actions and reasoning
        raw_candidates = [action for action, _ in raw_candidates_with_reasoning]
        raw_reasoning = [reasoning for _, reasoning in raw_candidates_with_reasoning]
        
        candidates = self._dedupe_preserve(raw_candidates)
        
        if not candidates:
            action, reasoning = self._call_reasoner(reasoner, messages, obs, plan, instruction_prompt)
            candidates = [action]
            reasoning_texts = [reasoning]
        else:
            # Map deduped candidates back to their reasoning
            reasoning_texts = []
            for candidate in candidates:
                for original_action, original_reasoning in raw_candidates_with_reasoning:
                    if original_action == candidate:
                        reasoning_texts.append(original_reasoning)
                        break
        
        scores = [self._score(a, messages, obs, plan, prm_scoring_prompt) for a in candidates]
        
        order = list(range(len(candidates)))
        self.rng.shuffle(order)
        best_idx = max(order, key=lambda i: scores[i])
        best_action = candidates[best_idx]
        best_reasoning = reasoning_texts[best_idx]

        metadata = {
            "strategy": "best_of_n",
            "compute_config": dict(config),
            "num_reasoner_calls": len(raw_candidates) if raw_candidates else 1,
            "num_scored": len(candidates),
            "candidates": candidates,
            "reasoning_texts": reasoning_texts,
            "scores": scores,
            "chosen_index": best_idx,
            "chosen_reasoning": best_reasoning,
            "deduped": len(candidates) != len(raw_candidates),
        }
        return best_action, metadata


class BeamSearchStrategy(ComputeStrategy):
    
    def execute(self, reasoner, messages, obs, plan, config, instruction_prompt, prm_scoring_prompt):
        """
        Beam search strategy: iterative refinement with beam width.
        Returns: (best_action, metadata) where metadata includes all reasoning texts
        """
        beam_width = int(config.get("param", 2))
        beam_width = max(2, min(64, beam_width))
        rounds = int(config.get("rounds", 2))
        rounds = max(1, min(8, rounds))
        
        total_calls = 0
        
        # Initial seed generation
        seed_with_reasoning = [
            self._call_reasoner(reasoner, messages, obs, plan, instruction_prompt) 
            for _ in range(beam_width * 2)
        ]
        total_calls += len(seed_with_reasoning)
        
        seed = [action for action, _ in seed_with_reasoning]
        seed_reasoning = [reasoning for _, reasoning in seed_with_reasoning]
        
        seed = self._dedupe_preserve(seed)
        seed_scores = [self._score(a, messages, obs, plan, prm_scoring_prompt) for a in seed]
        idx = sorted(range(len(seed)), key=lambda i: seed_scores[i], reverse=True)[:beam_width]
        beam = [seed[i] for i in idx]
        beam_scores = [seed_scores[i] for i in idx]
        beam_reasoning = [seed_reasoning[i] for i in idx]
        
        # Iterative beam expansion
        for _ in range(rounds - 1):
            expanded_with_reasoning = [
                self._call_reasoner(reasoner, messages, obs, plan, instruction_prompt) 
                for _a in beam
            ]
            total_calls += len(expanded_with_reasoning)
            
            expanded = [action for action, _ in expanded_with_reasoning]
            expanded_reasoning = [reasoning for _, reasoning in expanded_with_reasoning]
            
            pool = self._dedupe_preserve(beam + expanded)
            pool_scores = [self._score(a, messages, obs, plan, prm_scoring_prompt) for a in pool]
            
            # Map reasoning back to pool items
            pool_reasoning = []
            for item in pool:
                for original_action, original_reasoning in zip(beam + expanded, beam_reasoning + expanded_reasoning):
                    if original_action == item:
                        pool_reasoning.append(original_reasoning)
                        break
            
            idx = sorted(range(len(pool)), key=lambda i: pool_scores[i], reverse=True)[:beam_width]
            beam = [pool[i] for i in idx]
            beam_scores = [pool_scores[i] for i in idx]
            beam_reasoning = [pool_reasoning[i] for i in idx]
        
        best_idx = max(range(len(beam)), key=lambda i: beam_scores[i])
        best_action = beam[best_idx]
        best_reasoning = beam_reasoning[best_idx]
        
        metadata = {
            "strategy": "beam_search",
            "compute_config": dict(config),
            "num_reasoner_calls": total_calls,
            "beam_width": beam_width,
            "rounds": rounds,
            "beam": beam,
            "beam_reasoning": beam_reasoning,
            "beam_scores": beam_scores,
            "chosen_index": best_idx,
            "chosen_reasoning": best_reasoning,
        }
        return best_action, metadata


class LookaheadStrategy(ComputeStrategy):
    
    def execute(self, reasoner, messages, obs, plan, config, instruction_prompt, prm_scoring_prompt):
        """
        Lookahead strategy: k-step rollouts from seed actions.
        Returns: (best_action, metadata) where metadata includes all reasoning texts
        """
        k = int(config.get("param", 1))
        k = max(1, min(4, k))
        seeds = int(config.get("seeds", 4))
        seeds = max(2, min(32, seeds))
        
        total_calls = 0
        
        seed_with_reasoning = [
            self._call_reasoner(reasoner, messages, obs, plan, instruction_prompt) 
            for _ in range(seeds)
        ]
        total_calls += len(seed_with_reasoning)
        
        seed_actions = [action for action, _ in seed_with_reasoning]
        seed_reasoning = [reasoning for _, reasoning in seed_with_reasoning]
        seed_actions = self._dedupe_preserve(seed_actions)
        
        per_seed_best = []
        all_rollouts = []
        
        for seed_idx, a in enumerate(seed_actions):
            rollouts_with_reasoning = [
                self._call_reasoner(reasoner, messages, obs, plan, instruction_prompt) 
                for _ in range(k)
            ]
            total_calls += len(rollouts_with_reasoning)
            
            rollouts = [action for action, _ in rollouts_with_reasoning]
            rollouts_reasoning = [reasoning for _, reasoning in rollouts_with_reasoning]
            rollouts = self._dedupe_preserve(rollouts)
            
            if rollouts:
                scores = [self._score(r, messages, obs, plan, prm_scoring_prompt) for r in rollouts]
            else:
                scores = [self._score(a, messages, obs, plan, prm_scoring_prompt)]
                rollouts = [a]
                rollouts_reasoning = [seed_reasoning[seed_idx] if seed_idx < len(seed_reasoning) else ""]
            
            local_idx = max(range(len(rollouts)), key=lambda i: scores[i])
            chosen_rollout = rollouts[local_idx]
            chosen_rollout_reasoning = rollouts_reasoning[local_idx]


            per_seed_best.append((a, chosen_rollout, scores[local_idx], chosen_rollout_reasoning))
            all_rollouts.append({
                "seed": a, 
                "seed_reasoning": seed_reasoning[seed_idx] if seed_idx < len(seed_reasoning) else "",
                "rollouts": rollouts, 
                "rollouts_reasoning": rollouts_reasoning,
                "scores": scores, 
                "chosen": local_idx,
                "chosen_reasoning": chosen_rollout_reasoning,
            })
        
        global_idx = max(range(len(per_seed_best)), key=lambda i: per_seed_best[i][2])
        chosen_seed, chosen_rollout, chosen_score, chosen_reasoning = per_seed_best[global_idx]
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
            "chosen_reasoning": chosen_reasoning,
        }
        return best_action, metadata


def get_compute_strategy(strategy_name, prm_model, rng=None):
    """Return the compute strategy instance for `strategy_name`."""
    strategies = {
        "best_of_n": BestOfNStrategy,
        "beam_search": BeamSearchStrategy,
        "lookahead": LookaheadStrategy,
    }
    cls = strategies.get(strategy_name, BestOfNStrategy)
    return cls(prm_model, rng=rng)
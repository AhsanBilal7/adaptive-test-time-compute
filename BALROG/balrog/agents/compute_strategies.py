"""PRM-guided compute strategies (Best-of-N, beam search, lookahead) for BALROG agents."""

from typing import Any, Dict, List, Tuple
import random


class ComputeStrategy:
    
    def __init__(self, prm_model, rng=None):
        self.prm_model = prm_model
        self.rng = rng if rng is not None else random.Random(0)
    
    # ---- utilities ----------------------------------------------------------
    def _call_reasoner(self, reasoner, messages, obs, plan) -> str:
        """
        Normalizes different reasoner interfaces to a single action string.
        """
        # heuristic-script often uses (obs, plan)
        if hasattr(reasoner, "generate_action"):
            try:
                # try messages,plan signature first
                out = reasoner.generate_action(messages, plan)
            except TypeError:
                out = reasoner.generate_action(obs, plan)
        else:
            raise RuntimeError("Reasoner has no generate_action(...)")

        # Many reasoners return (action, extra)
        if isinstance(out, tuple) and len(out) >= 1:
            action = out[0]
        else:
            action = out
        return str(action).strip()

    def _score(self, action: str, messages, obs, plan) -> float:
        """
        Safe PRM scoring with clamping and fallbacks.
        """
        try:
            score = self.prm_model.score_response(action, obs, plan, context="")
        except Exception:
            score = 0.0
        try:
            s = float(score)
        except Exception:
            s = 0.0
        if s != s:  # NaN
            s = 0.0
        return max(0.0, min(1.0, s))

    def _dedupe_preserve(self, actions: List[str]) -> List[str]:
        seen = set()
        out = []
        for a in actions:
            if a not in seen:
                seen.add(a)
                out.append(a)
        return out

    # ---- public interface ---------------------------------------------------
    def execute(self, reasoner, messages, obs, plan, config: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        raise NotImplementedError


class BestOfNStrategy(ComputeStrategy):
    def execute(self, reasoner, messages, obs, plan, config):
        n = int(config.get("param", 1))
        n = max(1, min(1024, n))

        raw_candidates = [self._call_reasoner(reasoner, messages, obs, plan) for _ in range(n)]
        candidates = self._dedupe_preserve(raw_candidates)
        # Ensure we evaluate at least one
        if not candidates:
            candidates = [self._call_reasoner(reasoner, messages, obs, plan)]

        scores = [self._score(a, messages, obs, plan) for a in candidates]

        # stable, seeded tie-break
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
    """
    Lightweight textual beam: sample (approx.) beam_width * R candidates,
    score with PRM, keep top beam_width, iterate R=2 rounds (configurable).
    This acts as a conservative, PRM-guided diversification vs pure best-of-N.
    """
    def execute(self, reasoner, messages, obs, plan, config):
        beam_width = int(config.get("param", 2))
        beam_width = max(2, min(64, beam_width))
        rounds = int(config.get("rounds", 2))
        rounds = max(1, min(8, rounds))

        total_calls = 0
        beam: List[str] = []

        # round 0: seed
        seed = [self._call_reasoner(reasoner, messages, obs, plan) for _ in range(beam_width * 2)]
        total_calls += len(seed)
        seed = self._dedupe_preserve(seed)
        seed_scores = [self._score(a, messages, obs, plan) for a in seed]
        # take top beam_width
        idx = sorted(range(len(seed)), key=lambda i: seed_scores[i], reverse=True)[:beam_width]
        beam = [seed[i] for i in idx]
        beam_scores = [seed_scores[i] for i in idx]

        # refinement rounds
        for _ in range(rounds - 1):
            # expand each beam element with one variant
            expanded = []
            for _a in beam:
                expanded.append(self._call_reasoner(reasoner, messages, obs, plan))
            total_calls += len(expanded)
            pool = self._dedupe_preserve(beam + expanded)
            pool_scores = [self._score(a, messages, obs, plan) for a in pool]
            idx = sorted(range(len(pool)), key=lambda i: pool_scores[i], reverse=True)[:beam_width]
            beam = [pool[i] for i in idx]
            beam_scores = [pool_scores[i] for i in idx]

        # pick best in final beam
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
    """
    Pragmatic lookahead: for each of M seeds, generate k rollouts (k=param),
    score the rollout endpoints via PRM and pick the seed with highest max rollout score.
    This respects budget and avoids exponential blow-ups.
    """
    def execute(self, reasoner, messages, obs, plan, config):
        k = int(config.get("param", 1))
        k = max(1, min(4, k))
        seeds = int(config.get("seeds", 4))
        seeds = max(2, min(32, seeds))

        total_calls = 0

        # generate seed actions
        seed_actions = [self._call_reasoner(reasoner, messages, obs, plan) for _ in range(seeds)]
        total_calls += len(seed_actions)
        seed_actions = self._dedupe_preserve(seed_actions)

        # for each seed, simulate k rollouts (new actions) and score them
        per_seed_best = []
        all_rollouts = []

        for a in seed_actions:
            rollouts = [self._call_reasoner(reasoner, messages, obs, plan) for _ in range(k)]
            total_calls += len(rollouts)
            rollouts = self._dedupe_preserve(rollouts)
            scores = [self._score(r, messages, obs, plan) for r in rollouts] or [self._score(a, messages, obs, plan)]
            # fallback if deduped to empty
            if not rollouts:
                rollouts = [a]
            # pick the best rollout for this seed
            local_idx = max(range(len(rollouts)), key=lambda i: scores[i])
            per_seed_best.append((a, rollouts[local_idx], scores[local_idx]))
            all_rollouts.append({"seed": a, "rollouts": rollouts, "scores": scores, "chosen": local_idx})

        # choose the seed with the strongest reachable rollout
        global_idx = max(range(len(per_seed_best)), key=lambda i: per_seed_best[i][2])
        chosen_seed, chosen_rollout, chosen_score = per_seed_best[global_idx]
        # Output the *rollout* as the action; alternatively, return seed.
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


def get_compute_strategy(strategy_name: str, prm_model, rng=None):
    """
    Factory function to get compute strategy.
    
    Args:
        strategy_name: Name of strategy ('best_of_n', 'beam_search', 'lookahead')
        prm_model: PRM model instance
        rng: Random number generator for deterministic tie-breaking
        
    Returns:
        ComputeStrategy: Strategy instance
    """
    strategies = {
        "best_of_n": BestOfNStrategy,
        "beam_search": BeamSearchStrategy,
        "lookahead": LookaheadStrategy,
    }
    
    cls = strategies.get(strategy_name, BestOfNStrategy)
    return cls(prm_model, rng=rng)
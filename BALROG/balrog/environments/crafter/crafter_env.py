from typing import Optional

import crafter
import gymnasium as gym
from balrog.environments.crafter import CrafterLanguageWrapper
from balrog.environments.wrappers import GymV21CompatibilityV0


# ---- fallback shim: works on any Gymnasium version ----
def _wrap_env_compat(env):
    """Convert a legacy gym.Env into something gymnasium can wrap."""
    try:
        import gym as legacy_gym
    except ImportError:
        legacy_gym = None

    # If it's already Gymnasium-compatible, leave it alone
    if isinstance(env, gym.Env):
        return env
    
    # If it's a legacy gym.Env, wrap minimally
    if legacy_gym is not None and isinstance(env, legacy_gym.Env):
        class GymCompat(gym.Env):
            metadata = getattr(env, "metadata", {})
            render_mode = getattr(env, "render_mode", None)

            def __init__(self, e):
                self.env = e
                # Expose spaces so downstream wrappers can access them
                self.observation_space = getattr(e, "observation_space", None)
                self.action_space = getattr(e, "action_space", None)
                self.reward_range = getattr(e, "reward_range", (-float("inf"), float("inf")))

            def reset(self, *args, **kwargs):
                obs = self.env.reset()
                # Return obs, info pair for Gymnasium API
                return obs, {}

            def step(self, action):
                out = self.env.step(action)
                
                # Normalize whatever the underlying env returns
                if len(out) == 5:
                    # Gymnasium-style: (obs, reward, terminated, truncated, info)
                    obs, reward, terminated, truncated, info = out
                    done = terminated or truncated
                    return obs, reward, done, info
                elif len(out) == 4:
                    # Legacy Gym-style: (obs, reward, done, info)
                    return out
                else:
                    raise ValueError(f"Unexpected number of return values from env.step(): {len(out)}")

            def render(self, *a, **kw):
                return self.env.render(*a, **kw)

            def close(self):
                return self.env.close()

        return GymCompat(env)
    
    # If neither gym nor gymnasium applies, return unchanged
    return env
# ---------------------------------------------------------


def make_crafter_env(env_name, task, config, render_mode: Optional[str] = None):
    crafter_kwargs = dict(config.envs.crafter_kwargs)
    max_episode_steps = crafter_kwargs.pop("max_episode_steps", 2)
    unique_items = crafter_kwargs.pop("unique_items", True)
    precise_location = crafter_kwargs.pop("precise_location", False)
    skip_items = crafter_kwargs.pop("skip_items", [])
    edge_only_items = crafter_kwargs.pop("edge_only_items", [])

    for param in ["area", "view", "size"]:
        if param in crafter_kwargs:
            crafter_kwargs[param] = tuple(crafter_kwargs[param])

    # Create raw Crafter env
    env = crafter.Env(**crafter_kwargs)
    
    # ✅ Wrap for Gymnasium compatibility using fallback shim
    env = _wrap_env_compat(env)
    
    # Now safe to wrap with custom language wrapper
    env = CrafterLanguageWrapper(
        env,
        task,
        max_episode_steps=max_episode_steps,
        unique_items=unique_items,
        precise_location=precise_location,
        skip_items=skip_items,
        edge_only_items=edge_only_items,
    )
    env = GymV21CompatibilityV0(env=env, render_mode=render_mode)

    return env

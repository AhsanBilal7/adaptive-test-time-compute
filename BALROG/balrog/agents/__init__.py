from balrog.client import create_llm_client
from hydra.utils import instantiate

from ..prompt_builder import create_prompt_builder
from .chain_of_thought import ChainOfThoughtAgent
from .custom import CustomAgent
from .dummy import DummyAgent
from .few_shot import FewShotAgent
from .naive import NaiveAgent
from .robust_naive import RobustNaiveAgent
from .robust_cot import RobustCoTAgent


class AgentFactory:
    """Factory class for creating agents based on configuration.

    The `AgentFactory` class is responsible for initializing the appropriate agent type
    based on the provided configuration, which includes setting up the LLM client and
    prompt builder.
    
    Supports two modes:
    1. Legacy mode: Uses `type` field with hardcoded string matching
    2. Hydra mode: Uses `_target_` field for flexible class instantiation
    """

    def __init__(self, config):
        """Initialize the AgentFactory with configuration settings.

        Args:
            config (omegaconf.DictConfig): Configuration object containing settings for the agent and client.
        """
        self.config = config

    def create_agent(self):
        """Create an agent instance based on the agent type specified in the configuration.

        The function supports two modes:
        1. If `_target_` is specified in config.agent, use Hydra's instantiate for flexible class loading
        2. Otherwise, fall back to legacy `type` attribute for backward compatibility

        Returns:
            Agent: An instance of the selected agent type, configured with the client and prompt builder.

        Raises:
            ValueError: If an unknown agent type is specified in the configuration.
        """
        client_factory = create_llm_client(self.config.client)
        prompt_builder = create_prompt_builder(self.config.agent)

        # Check if using Hydra's _target_ pattern for flexible instantiation
        if hasattr(self.config.agent, '_target_') and self.config.agent._target_ is not None:
            # Extract only CustomAgent-specific parameters from config
            agent_params = {
                'client_factory': client_factory,
                'prompt_builder': prompt_builder,
            }
            
            # Add optional parameters if present in config
            if hasattr(self.config.agent, 'mode'):
                agent_params['mode'] = self.config.agent.mode
            if hasattr(self.config.agent, 'planning_frequency'):
                agent_params['planning_frequency'] = self.config.agent.planning_frequency
            if hasattr(self.config.agent, 'use_planner'):
                agent_params['use_planner'] = self.config.agent.use_planner
            if hasattr(self.config.agent, 'use_tool_selector'):
                agent_params['use_tool_selector'] = self.config.agent.use_tool_selector
            if hasattr(self.config.agent, 'use_compute_selector'):
                agent_params['use_compute_selector'] = self.config.agent.use_compute_selector
            if hasattr(self.config.agent, 'fixed_tool'):
                agent_params['fixed_tool'] = self.config.agent.fixed_tool
            if hasattr(self.config.agent, 'fixed_compute'):
                agent_params['fixed_compute'] = self.config.agent.fixed_compute
            if hasattr(self.config.agent, 'dataset'):
                agent_params['dataset'] = self.config.agent.dataset
            
            # Use Hydra's instantiate with filtered parameters
            agent = instantiate(
                {'_target_': self.config.agent._target_},
                **agent_params
            )
            print(f"[DEBUG] ✅ Agent instantiated via Hydra _target_: {self.config.agent._target_}")
            return agent

        # Legacy mode: Use type string for backward compatibility
        if self.config.agent.type == "naive":
            return NaiveAgent(client_factory, prompt_builder)
        elif self.config.agent.type == "cot":
            return ChainOfThoughtAgent(client_factory, prompt_builder, config=self.config)
        elif self.config.agent.type == "dummy":
            return DummyAgent(client_factory, prompt_builder)
        elif self.config.agent.type == "custom":
            return CustomAgent(client_factory, prompt_builder)
        elif self.config.agent.type == "few_shot":
            return FewShotAgent(client_factory, prompt_builder, self.config.agent.max_icl_history)
        elif self.config.agent.type == "robust_naive":
            return RobustNaiveAgent(client_factory, prompt_builder)
        elif self.config.agent.type == "robust_cot":
            return RobustCoTAgent(client_factory, prompt_builder, config=self.config)

        else:
            raise ValueError(f"Unknown agent type: {self.config.agent}")

import base64
import datetime
import logging
import time
import json
import csv
import os
from collections import namedtuple
from io import BytesIO

import google.generativeai as genai
from anthropic import Anthropic
from google.generativeai import caching
from openai import OpenAI

LLMResponse = namedtuple(
    "LLMResponse",
    [
        "model_id",
        "completion",
        "stop_reason",
        "input_tokens",
        "output_tokens",
        "reasoning",
    ],
)

httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class LLMClientWrapper:
    """Base class for LLM client wrappers.

    Provides common functionality for interacting with different LLM APIs, including
    handling retries and common configuration settings. Subclasses should implement
    the `generate` method specific to their LLM API.
    """

    def __init__(self, client_config):
        """Initialize the LLM client wrapper with configuration settings.

        Args:
            client_config: Configuration object containing client-specific settings.
        """
        self.client_name = client_config.client_name
        self.model_id = client_config.model_id
        self.base_url = client_config.base_url
        self.timeout = client_config.timeout
        self.client_kwargs = {**client_config.generate_kwargs}
        self.max_retries = client_config.max_retries
        self.delay = client_config.delay
        self.alternate_roles = client_config.alternate_roles

    def generate(self, messages):
        """Generate a response from the LLM given a list of messages.

        This method should be overridden by subclasses.

        Args:
            messages (list): A list of messages to send to the LLM.

        Returns:
            LLMResponse: The response from the LLM.
        """
        raise NotImplementedError("This method should be overridden by subclasses")

    def generate_with_structured(self, messages, schema):
        """Generate a structured JSON response from the LLM given a list of messages and a schema.

        This method should be overridden by subclasses.

        Args:
            messages (list): A list of messages to send to the LLM.
            schema (dict): JSON schema defining the expected output structure.

        Returns:
            LLMResponse: The response from the LLM with JSON content.
        """
        raise NotImplementedError("This method should be overridden by subclasses")

    def execute_with_retries(self, func, *args, **kwargs):
        """Execute a function with retries upon failure.

        Args:
            func (callable): The function to execute.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            Any: The result of the function call.

        Raises:
            Exception: If the function fails after the maximum number of retries.
        """
        retries = 0
        while retries < self.max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                retries += 1
                logger.error(f"Retryable error during {func.__name__}: {e}. Retry {retries}/{self.max_retries}")
                sleep_time = self.delay * (2 ** (retries - 1))  # Exponential backoff
                time.sleep(sleep_time)
        raise Exception(f"Failed to execute {func.__name__} after {self.max_retries} retries.")


def process_image_openai(image):
    """Process an image for OpenAI API by converting it to base64.

    Args:
        image: The image to process.

    Returns:
        dict: A dictionary containing the image data formatted for OpenAI.
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    # Return the image content for OpenAI
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
    }


def process_image_claude(image):
    """Process an image for Anthropic's Claude API by converting it to base64.

    Args:
        image: The image to process.

    Returns:
        dict: A dictionary containing the image data formatted for Claude.
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    # Return the image content for Anthropic
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": base64_image},
    }


class OpenAIWrapper(LLMClientWrapper):
    """Wrapper for interacting with the OpenAI API."""

    def __init__(self, client_config):
        """Initialize the OpenAIWrapper with the given configuration.

        Args:
            client_config: Configuration object containing client-specific settings.
        """
        super().__init__(client_config)
        self._initialized = False

    def _initialize_client(self):
        """Initialize the OpenAI client if not already initialized."""
        if not self._initialized:
            if self.client_name.lower() == "vllm":
                self.client = OpenAI(api_key="EMPTY", base_url=self.base_url)
            elif self.client_name.lower() == "nvidia" or self.client_name.lower() == "xai":
                if not self.base_url or not self.base_url.strip():
                    raise ValueError("base_url must be provided when using NVIDIA or XAI client")
                self.client = OpenAI(base_url=self.base_url)
            elif self.client_name.lower() == "openai":
                # For OpenAI, always use the standard API regardless of base_url
                self.client = OpenAI()
            self._initialized = True

    def convert_messages(self, messages):
        """Convert messages to the format expected by the OpenAI API.

        Args:
            messages (list): A list of message objects.

        Returns:
            list: A list of messages formatted for the OpenAI API.
        """
        converted_messages = []
        for msg in messages:
            new_content = [{"type": "text", "text": msg.content}]
            if msg.attachment is not None:
                new_content.append(process_image_openai(msg.attachment))
            if self.alternate_roles and converted_messages and converted_messages[-1]["role"] == msg.role:
                converted_messages[-1]["content"].extend(new_content)
            else:
                converted_messages.append({"role": msg.role, "content": new_content})
        return converted_messages

    def generate(self, messages):
        """Generate a response from the OpenAI API given a list of messages.

        Args:
            messages (list): A list of message objects.

        Returns:
            LLMResponse: The response from the OpenAI API.
        """
        self._initialize_client()
        converted_messages = self.convert_messages(messages)

        def api_call():
            # Create kwargs for the API call
            api_kwargs = {
                "messages": converted_messages,
                "model": self.model_id,
                "max_tokens": self.client_kwargs.get("max_tokens", 1024),
            }

            # Only include temperature if it's not None
            temperature = self.client_kwargs.get("temperature")
            if temperature is not None:
                api_kwargs["temperature"] = temperature

            return self.client.chat.completions.create(**api_kwargs)

        response = self.execute_with_retries(api_call)

        return LLMResponse(
            model_id=self.model_id,
            completion=response.choices[0].message.content.strip(),
            stop_reason=response.choices[0].finish_reason,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            reasoning=None,
        )

    def generate_with_structured(self, messages, schema):
        """Generate a structured JSON response from the OpenAI API.

        Args:
            messages (list): A list of message objects.
            schema (dict): JSON schema defining the expected output structure.

        Returns:
            LLMResponse: The response from the OpenAI API with JSON content.
        """
        self._initialize_client()
        converted_messages = self.convert_messages(messages)

        def api_call():
            # Create kwargs for the API call
            api_kwargs = {
                "messages": converted_messages,
                "model": self.model_id,
                "max_tokens": self.client_kwargs.get("max_tokens", 1024),
            }

            # Only include temperature if it's not None
            temperature = self.client_kwargs.get("temperature")
            if temperature is not None:
                api_kwargs["temperature"] = temperature

            # Add structured output configuration based on client
            if self.client_name.lower() in ["openai", "nvidia", "xai"]:
                # Use response_format for OpenAI-style structured outputs
                api_kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "strict": True,
                        "schema": schema
                    }
                }
            else:
                # For vLLM, use simpler JSON mode with schema in prompt
                api_kwargs["response_format"] = {"type": "json_object"}
                # Add schema to system message
                schema_instruction = f"\n\nYou must respond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"
                if converted_messages and converted_messages[0]["role"] == "system":
                    converted_messages[0]["content"][0]["text"] += schema_instruction
                else:
                    converted_messages.insert(0, {
                        "role": "system",
                        "content": [{"type": "text", "text": f"You are a helpful assistant that responds in JSON format.{schema_instruction}"}]
                    })
                api_kwargs["messages"] = converted_messages

            return self.client.chat.completions.create(**api_kwargs)

        response = self.execute_with_retries(api_call)

        return LLMResponse(
            model_id=self.model_id,
            completion=response.choices[0].message.content.strip(),
            stop_reason=response.choices[0].finish_reason,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            reasoning=None,
        )


class OllamaWrapper(LLMClientWrapper):
    """Wrapper for interacting with Ollama's API using native ollama library.
    
    Uses the native Python ollama library for better integration and structured outputs.
    Supports both OpenAI-compatible API and native ollama library methods.
    """

    def __init__(self, client_config):
        """Initialize the OllamaWrapper with the given configuration.

        Args:
            client_config: Configuration object containing client-specific settings.
        """
        super().__init__(client_config)
        self._initialized = False
        self._use_native = False  # Flag to determine which API to use

    def _initialize_client(self):
        """Initialize the Ollama client if not already initialized."""
        if not self._initialized:
            # Try to use native ollama library first
            try:
                import ollama
                
                self.ollama = ollama
                self._use_native = True
                
                # Check if Ollama daemon is running
                if not self._is_ollama_available():
                    logger.warning("Ollama daemon not reachable. Falling back to OpenAI-compatible API.")
                    self._use_native = False
                else:
                    # Ensure model is available
                    self._ensure_model()
                    self._initialized = True
                    return
                    
            except ImportError:
                logger.debug("Native ollama library not found. Using OpenAI-compatible API.")
                self._use_native = False
            except Exception as e:
                logger.warning(f"Error initializing native ollama: {e}. Falling back to OpenAI-compatible API.")
            
            # Fallback to OpenAI-compatible API
            if not self.base_url or not self.base_url.strip():
                self.base_url = "http://localhost:11434/v1"
            

    def _is_ollama_available(self) -> bool:
        """Check if Ollama daemon is running."""
        import socket
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex(('127.0.0.1', 11434))
            sock.close()
            if result == 0:
                return True
        except Exception:
            pass
        
        # Try listing models as backup check
        try:
            self.ollama.list()
            return True
        except Exception:
            pass
        
        return False

    def _ensure_model(self):
        """Pull model if not already available."""
        try:
            models = self.ollama.list()
            available = {m["model"] for m in models.get("models", [])}
            
            # Check if exact model name exists
            if self.model_id in available:
                return
            
            # Check if model exists with :latest suffix
            model_base = self.model_id.split(':')[0]
            for model in available:
                if model.startswith(model_base):
                    logger.info(f"Using existing model: {model}")
                    self.model_id = model  # Use the available variant
                    return
            
            # Model not found, try to pull it
            logger.info(f"Model {self.model_id} not found locally. Pulling...")
            for _ in self.ollama.pull(self.model_id, stream=True):
                pass
            logger.info(f"Model {self.model_id} ready.")
            
        except Exception as e:
            logger.warning(f"Could not verify model availability: {e}")
            logger.info(f"Will attempt to use model anyway: {self.model_id}")

    def _build_options(self) -> dict:
        """Build options dict for native Ollama API."""
        opts = {
            "temperature": self.client_kwargs.get("temperature", 0.7),
            "num_ctx": self.client_kwargs.get("max_tokens", 1024)
        }
        
        if "seed" in self.client_kwargs and self.client_kwargs["seed"] is not None:
            opts["seed"] = self.client_kwargs["seed"]
        
        return opts

    def convert_messages(self, messages):
        """Convert messages to the format expected by the Ollama API.

        Args:
            messages (list): A list of message objects or dictionaries.

        Returns:
            list: A list of messages formatted for the Ollama API.
        """
        converted_messages = []
        
        for msg in messages:
            # Handle both dict and object formats
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                attachment = msg.get("attachment")
            else:
                role = getattr(msg, "role", "user")
                content = getattr(msg, "content", "")
                attachment = getattr(msg, "attachment", None)
            
            if self._use_native:
                # Native ollama format - simpler
                converted_messages.append({
                    "role": role,
                    "content": content
                })
                # Note: Native ollama doesn't use the image format in messages
                # Images are passed separately in the API call
            else:
                # OpenAI-compatible format
                new_content = [{"type": "text", "text": content}]
                
                if attachment is not None:
                    new_content.append(process_image_openai(attachment))
                
                # Merge with previous message if same role and alternate_roles is True
                if self.alternate_roles and converted_messages and converted_messages[-1]["role"] == role:
                    if isinstance(converted_messages[-1]["content"], list):
                        converted_messages[-1]["content"].extend(new_content)
                    else:
                        converted_messages[-1]["content"] = [
                            {"type": "text", "text": converted_messages[-1]["content"]}
                        ] + new_content
                else:
                    converted_messages.append({"role": role, "content": new_content})
                
        return converted_messages

    def _extract_system_and_user_messages(self, messages):
        """Extract system and user messages from message list.
        
        Returns messages in format [system_messages] + [conversation_messages]
        where system messages come first.
        
        Args:
            messages (list): List of message dicts with 'role' and 'content'
            
        Returns:
            tuple: (system_messages, other_messages)
        """
        system_messages = []
        other_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_messages.append(msg)
            else:
                other_messages.append(msg)
        
        return system_messages, other_messages

    def generate(self, messages):
        """Generate a response from the Ollama API given a list of messages.

        Args:
            messages (list): A list of message objects or dictionaries.

        Returns:
            LLMResponse: The response from the Ollama API.
        """
        self._initialize_client()
        converted_messages = self.convert_messages(messages)


        response = self.ollama.chat(
            model=self.model_id,
            messages=converted_messages,
            options=self._build_options(),
            keep_alive="30m"  # Keep model loaded for 30 minutes
        )
        

        # stream=False,
        # keep_alive=self.client_kwargs.get("keep_alive", "5m")
        return LLMResponse(
            model_id=self.model_id,
            completion=response["message"]["content"].strip(),
            stop_reason="stop",  # Native API doesn't provide this
            input_tokens=response.get("prompt_eval_count", 0),
            output_tokens=response.get("eval_count", 0),
            reasoning=None,
        )


    def generate_with_structured(self, messages, schema):
        """Generate a structured JSON response from the Ollama API.

        Messages should be in format [system_message] + [user_message] for best results.
        The schema is passed directly to Ollama's format parameter for native structured outputs.

        Args:
            messages (list): A list of message objects or dictionaries.
                           Format: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
            schema (dict): JSON schema defining the expected output structure.

        Returns:
            LLMResponse: The response from the Ollama API with JSON content.
        """
        self._initialize_client()
        converted_messages = self.convert_messages(messages)

        # if self._use_native:
        # Use native ollama library with structured output
        # Separate system and other messages
        system_messages, other_messages = self._extract_system_and_user_messages(converted_messages)
        
        # Combine: system messages first, then the rest
        final_messages = system_messages + other_messages
        # print("====================================================")
        # print(final_messages)
        # print("====================================================")

        response = self.ollama.chat(
            model=self.model_id,
            messages=final_messages,
            format=schema,  # Pass schema directly to format parameter
            options=self._build_options(),
            keep_alive="30m"  # Keep model loaded for 30 minutes
        )
        
        # stream=False,
        # keep_alive=self.client_kwargs.get("keep_alive", "5m")
        return LLMResponse(
            model_id=self.model_id,
            completion=response["message"]["content"].strip(),
            stop_reason="stop",
            input_tokens=response.get("prompt_eval_count", 0),
            output_tokens=response.get("eval_count", 0),
            reasoning=None,
        )

class GoogleGenerativeAIWrapper(LLMClientWrapper):
    """Wrapper for interacting with Google's Generative AI API."""

    def __init__(self, client_config):
        """Initialize the GoogleGenerativeAIWrapper with the given configuration.

        Args:
            client_config: Configuration object containing client-specific settings.
        """
        super().__init__(client_config)
        self._initialized = False

    def _initialize_client(self):
        """Initialize the Generative AI client if not already initialized."""
        if not self._initialized:
            self.model = genai.GenerativeModel(self.model_id)

            # Create kwargs dictionary for GenerationConfig
            client_kwargs = {
                "max_output_tokens": self.client_kwargs.get("max_tokens", 1024),
            }

            # Only include temperature if it's not None
            temperature = self.client_kwargs.get("temperature")
            if temperature is not None:
                client_kwargs["temperature"] = temperature

            self.generation_config = genai.types.GenerationConfig(**client_kwargs)
            self._initialized = True

    def convert_messages(self, messages):
        """Convert messages to the format expected by the Generative AI API.

        Args:
            messages (list): A list of message objects.

        Returns:
            list: A list of messages formatted for the Generative AI API.
        """
        # Convert standard Message objects to Gemini's format
        converted_messages = []
        for msg in messages:
            parts = []
            role = msg.role
            if role == "assistant":
                role = "model"
            elif role == "system":
                role = "user"
            if msg.content:
                parts.append(msg.content)
            if msg.attachment is not None:
                parts.append(msg.attachment)
            converted_messages.append(
                {
                    "role": role,
                    "parts": parts,
                }
            )
        return converted_messages

    def get_completion(self, converted_messages, max_retries=5, delay=5):
        """Get the completion from the model with retries upon failure.

        Args:
            converted_messages (list): Messages formatted for the Generative AI API.
            max_retries (int, optional): Maximum number of retries. Defaults to 5.
            delay (int, optional): Delay between retries in seconds. Defaults to 5.

        Returns:
            Response object from the API.

        Raises:
            Exception: If the API call fails after the maximum number of retries.
        """
        retries = 0
        while retries < max_retries:
            try:
                response = self.model.generate_content(
                    converted_messages,
                    generation_config=self.generation_config,
                )
                return response
            except Exception as e:
                retries += 1
                logger.error(f"Retryable error during generate_content: {e}. Retry {retries}/{max_retries}")
                sleep_time = delay * (2 ** (retries - 1))  # Exponential backoff
                time.sleep(sleep_time)

        # If maximum retries are reached and still no valid response
        raise Exception(f"Failed to get a valid completion after {max_retries} retries.")

    def extract_completion(self, response):
        """Extract the completion text from the API response.

        Args:
            response: The response object from the API.

        Returns:
            str: The extracted completion text.
            
        Raises:
            Exception: If response is None or missing expected fields.
        """
        if not response:
            raise Exception("Response is None, cannot extract completion.")

        candidates = getattr(response, "candidates", [])
        if not candidates:
            raise Exception("No candidates found in the response.")

        candidate = candidates[0]
        content = getattr(candidate, "content", None)
        if not content:
            raise Exception("No content found in the candidate.")
            
        content_parts = getattr(content, "parts", [])
        if not content_parts:
            raise Exception("No content parts found in the candidate.")

        text = getattr(content_parts[0], "text", None)
        if text is None:
            raise Exception("No text found in the content parts.")
            
        return text.strip()

    def generate(self, messages):
        """Generate a response from the Generative AI API given a list of messages.

        Args:
            messages (list): A list of message objects.

        Returns:
            LLMResponse: The response from the Generative AI API.
        """
        self._initialize_client()

        converted_messages = self.convert_messages(messages)

        def api_call():
            response = self.model.generate_content(
                converted_messages,
                generation_config=self.generation_config,
            )
            # Attempt to extract completion immediately after API call
            completion = self.extract_completion(response)
            # Return both response and completion if successful
            return response, completion

        try:
            # Execute the API call and extraction together with retries
            response, completion = self.execute_with_retries(api_call)

            # Check if the successful response contains an empty completion
            if not completion or completion.strip() == "":
                logger.warning(f"Gemini returned an empty completion for model {self.model_id}. Returning default empty response.")
                return LLMResponse(
                    model_id=self.model_id,
                    completion="",
                    stop_reason="empty_response",
                    input_tokens=getattr(response.usage_metadata, "prompt_token_count", 0) if response and getattr(response, "usage_metadata", None) else 0,
                    output_tokens=getattr(response.usage_metadata, "candidates_token_count", 0) if response and getattr(response, "usage_metadata", None) else 0,
                    reasoning=None,
                )
            else:
                # If completion is not empty, return the normal response
                return LLMResponse(
                    model_id=self.model_id,
                    completion=completion,
                    stop_reason=(
                        getattr(response.candidates[0], "finish_reason", "unknown")
                        if response and getattr(response, "candidates", [])
                        else "unknown"
                    ),
                    input_tokens=(
                        getattr(response.usage_metadata, "prompt_token_count", 0)
                        if response and getattr(response, "usage_metadata", None)
                        else 0
                    ),
                    output_tokens=(
                        getattr(response.usage_metadata, "candidates_token_count", 0)
                        if response and getattr(response, "usage_metadata", None)
                        else 0
                    ),
                    reasoning=None,
                )
        except Exception as e:
            logger.error(f"API call failed after {self.max_retries} retries: {e}. Returning empty completion.")
            # Return a default response indicating failure
            return LLMResponse(
                model_id=self.model_id,
                completion="",
                stop_reason="error_max_retries",
                input_tokens=0, # Assuming 0 tokens consumed if call failed
                output_tokens=0,
                reasoning=None,
            )

    def generate_with_structured(self, messages, schema):
        """Generate a structured JSON response from the Generative AI API.

        Args:
            messages (list): A list of message objects.
            schema (dict): JSON schema defining the expected output structure.

        Returns:
            LLMResponse: The response from the Generative AI API with JSON content.
        """
        self._initialize_client()
        converted_messages = self.convert_messages(messages)

        # Create generation config with JSON response format
        client_kwargs = {
            "max_output_tokens": self.client_kwargs.get("max_tokens", 1024),
            "response_mime_type": "application/json",
        }

        # Only include temperature if it's not None
        temperature = self.client_kwargs.get("temperature")
        if temperature is not None:
            client_kwargs["temperature"] = temperature

        # Add schema if supported
        if schema:
            client_kwargs["response_schema"] = schema

        generation_config = genai.types.GenerationConfig(**client_kwargs)

        def api_call():
            response = self.model.generate_content(
                converted_messages,
                generation_config=generation_config,
            )
            # Attempt to extract completion immediately after API call
            completion = self.extract_completion(response)
            # Return both response and completion if successful
            return response, completion

        try:
            # Execute the API call and extraction together with retries
            response, completion = self.execute_with_retries(api_call)

            # Check if the successful response contains an empty completion
            if not completion or completion.strip() == "":
                logger.warning(f"Gemini returned an empty completion for model {self.model_id}. Returning default empty response.")
                return LLMResponse(
                    model_id=self.model_id,
                    completion="{}",
                    stop_reason="empty_response",
                    input_tokens=getattr(response.usage_metadata, "prompt_token_count", 0) if response and getattr(response, "usage_metadata", None) else 0,
                    output_tokens=getattr(response.usage_metadata, "candidates_token_count", 0) if response and getattr(response, "usage_metadata", None) else 0,
                    reasoning=None,
                )
            else:
                # If completion is not empty, return the normal response
                return LLMResponse(
                    model_id=self.model_id,
                    completion=completion,
                    stop_reason=(
                        getattr(response.candidates[0], "finish_reason", "unknown")
                        if response and getattr(response, "candidates", [])
                        else "unknown"
                    ),
                    input_tokens=(
                        getattr(response.usage_metadata, "prompt_token_count", 0)
                        if response and getattr(response, "usage_metadata", None)
                        else 0
                    ),
                    output_tokens=(
                        getattr(response.usage_metadata, "candidates_token_count", 0)
                        if response and getattr(response, "usage_metadata", None)
                        else 0
                    ),
                    reasoning=None,
                )
        except Exception as e:
            logger.error(f"API call failed after {self.max_retries} retries: {e}. Returning empty JSON.")
            # Return a default response indicating failure
            return LLMResponse(
                model_id=self.model_id,
                completion="{}",
                stop_reason="error_max_retries",
                input_tokens=0,
                output_tokens=0,
                reasoning=None,
            )


class ClaudeWrapper(LLMClientWrapper):
    """Wrapper for interacting with Anthropic's Claude API."""

    def __init__(self, client_config):
        """Initialize the ClaudeWrapper with the given configuration.

        Args:
            client_config: Configuration object containing client-specific settings.
        """
        super().__init__(client_config)
        self._initialized = False

    def _initialize_client(self):
        """Initialize the Claude client if not already initialized."""
        if not self._initialized:
            self.client = Anthropic()
            self._initialized = True

    def convert_messages(self, messages):
        """Convert messages to the format expected by the Claude API.

        Args:
            messages (list): A list of message objects.

        Returns:
            list: A list of messages formatted for the Claude API.
        """
        converted_messages = []
        for msg in messages:
            converted_messages.append({"role": msg.role, "content": [{"type": "text", "text": msg.content}]})
            if converted_messages[-1]["role"] == "system":
                # Claude doesn't support system prompt and requires alternating roles
                converted_messages[-1]["role"] = "user"
                converted_messages.append({"role": "assistant", "content": "I'm ready!"})
            if msg.attachment is not None:
                converted_messages[-1]["content"].append(process_image_claude(msg.attachment))

        return converted_messages

    def generate(self, messages):
        """Generate a response from the Claude API given a list of messages.

        Args:
            messages (list): A list of message objects.

        Returns:
            LLMResponse: The response from the Claude API.
        """
        self._initialize_client()
        converted_messages = self.convert_messages(messages)

        def api_call():
            # Create kwargs for the API call
            api_kwargs = {
                "messages": converted_messages,
                "model": self.model_id,
                "max_tokens": self.client_kwargs.get("max_tokens", 1024),
            }

            # Only include temperature if it's not None
            temperature = self.client_kwargs.get("temperature")
            if temperature is not None:
                api_kwargs["temperature"] = temperature

            return self.client.messages.create(**api_kwargs)

        response = self.execute_with_retries(api_call)

        return LLMResponse(
            model_id=self.model_id,
            completion=response.content[0].text.strip(),
            stop_reason=response.stop_reason,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            reasoning=None,
        )

    def generate_with_structured(self, messages, schema):
        """Generate a structured JSON response from the Claude API.

        Note: Claude doesn't have native structured output support, so we use
        prompt engineering to request JSON format.

        Args:
            messages (list): A list of message objects.
            schema (dict): JSON schema defining the expected output structure.

        Returns:
            LLMResponse: The response from the Claude API with JSON content.
        """
        self._initialize_client()
        
        # Add JSON schema instruction to the last user message
        schema_instruction = f"\n\nPlease respond with valid JSON matching this schema:\n```json\n{json.dumps(schema, indent=2)}\n```\nRespond only with the JSON object, no additional text."
        
        # Create a modified messages list
        modified_messages = []
        for i, msg in enumerate(messages):
            if i == len(messages) - 1 and msg.role == "user":
                # Add schema instruction to last user message
                modified_msg = type(msg)(
                    role=msg.role,
                    content=msg.content + schema_instruction,
                    attachment=msg.attachment if hasattr(msg, 'attachment') else None
                )
                modified_messages.append(modified_msg)
            else:
                modified_messages.append(msg)
        
        converted_messages = self.convert_messages(modified_messages)

        def api_call():
            # Create kwargs for the API call
            api_kwargs = {
                "messages": converted_messages,
                "model": self.model_id,
                "max_tokens": self.client_kwargs.get("max_tokens", 1024),
            }

            # Only include temperature if it's not None
            temperature = self.client_kwargs.get("temperature")
            if temperature is not None:
                api_kwargs["temperature"] = temperature

            return self.client.messages.create(**api_kwargs)

        response = self.execute_with_retries(api_call)

        return LLMResponse(
            model_id=self.model_id,
            completion=response.content[0].text.strip(),
            stop_reason=response.stop_reason,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            reasoning=None,
        )


def create_llm_client(client_config):
    """
    Factory function to create the appropriate LLM client based on the client name.

    Args:
        client_config: Configuration object containing client-specific settings.

    Returns:
        callable: A factory function that returns an instance of the appropriate LLM client.
    """

    def client_factory():
        client_name_lower = client_config.client_name.lower()
        if "ollama" in client_name_lower:
            # Ollama has its own dedicated wrapper
            return OllamaWrapper(client_config)
        elif "openai" in client_name_lower or "vllm" in client_name_lower or "nvidia" in client_name_lower or "xai" in client_name_lower:
            # NVIDIA and XAI use OpenAI-compatible API, so we use the OpenAI wrapper
            return OpenAIWrapper(client_config)
        elif "gemini" in client_name_lower:
            return GoogleGenerativeAIWrapper(client_config)
        elif "claude" in client_name_lower:
            return ClaudeWrapper(client_config)
        else:
            raise ValueError(f"Unsupported client name: {client_config.client_name}")

    return client_factory
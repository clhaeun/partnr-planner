#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

from typing import Dict, List, Optional

import litellm
from omegaconf import DictConfig, OmegaConf

from habitat_llm.llm.base_llm import BaseLLM, Prompt
from habitat_llm.llm.openai_chat import generate_message


class LiteLLMChat(BaseLLM):
    """
    Unified LLM backend using LiteLLM.
    Supports 100+ providers (OpenAI, Anthropic, Google, etc.)
    via a single interface. Just change the model name:
      - "gpt-4o"                  -> OpenAI
      - "claude-sonnet-4-20250514"    -> Anthropic
      - "gemini/gemini-pro"       -> Google
    API keys are read from environment variables automatically
    (OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, etc.)
    """

    def __init__(self, conf: DictConfig):
        self.llm_conf = conf
        self.generation_params = self.llm_conf.generation_params
        self.verbose = self.llm_conf.verbose
        self.message_history: List[Dict] = []
        self.keep_message_history = self.llm_conf.keep_message_history

        litellm.drop_params = True

    def generate(
        self,
        prompt: Prompt,
        stop: Optional[str] = None,
        max_length: Optional[int] = None,
        generation_args=None,
        request_timeout: int = 40,
    ) -> str:
        """
        Generate a response using LiteLLM.
        :param prompt: A string or multimodal prompt (list of tuples).
        :param stop: A string that determines when to stop generation.
        :param max_length: The max number of tokens to generate.
        :param generation_args: Unused (constrained generation not supported).
        :param request_timeout: Maximum time before timeout.
        """
        params = OmegaConf.to_object(self.generation_params)

        if stop is None and params.get("stop"):
            stop = params["stop"]

        if max_length is not None:
            params["max_tokens"] = max_length

        messages = self.message_history.copy()
        if len(messages) == 0:
            messages.append({"role": "system", "content": self.llm_conf.system_message})

        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        else:
            messages.append(generate_message(prompt, image_detail="low"))

        kwargs = dict(
            model=params["model"],
            messages=messages,
            max_tokens=params.get("max_tokens", 250),
            temperature=params.get("temperature", 0),
            top_p=params.get("top_p", 1),
            stop=stop,
            timeout=request_timeout,
        )
        if params.get("api_base"):
            kwargs["api_base"] = params["api_base"]
        if params.get("api_key"):
            kwargs["api_key"] = params["api_key"]

        response = litellm.completion(**kwargs)
        text_response = response.choices[0].message.content
        self.response = text_response

        if self.keep_message_history:
            self.message_history = messages.copy()
            self.message_history.append({"role": "assistant", "content": text_response})

        if stop is not None and text_response:
            text_response = text_response.split(stop)[0]

        return text_response

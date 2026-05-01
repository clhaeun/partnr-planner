# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch

from habitat_llm.tools.motor_skills.skill import SkillPolicy
from habitat_llm.utils.grammar import FREE_TEXT


class OracleAskSkill(SkillPolicy):
    """
    A skill that allows the agent to ask the user a question via stdin.
    No simulator physical action is performed.
    """

    def __init__(
        self, config, observation_space, action_space, batch_size, env, agent_uid
    ):
        super().__init__(config, action_space, batch_size, False, agent_uid=agent_uid)
        self.env = env
        self._user_answer = ""

    def reset(self, batch_idxs):
        super().reset(batch_idxs)
        self._user_answer = ""

    def set_target(self, arg_string: str, env) -> None:
        """
        Display the agent's question and collect user's answer via stdin.
        """
        question = arg_string.strip()
        print(f"\n{'='*60}")
        print(f"[Agent asks]: {question}")
        print(f"{'='*60}")
        self._user_answer = input("[Your answer]: ").strip()

    def get_low_level_action(self, observations, deterministic=False):
        """
        Override: no simulator interaction needed.
        Return zero action tensor with user's answer.
        """
        actions = torch.zeros(
            self.prev_actions.shape, device=self.not_done_masks.device
        )
        actions = self._clip_actions(actions)

        msg = f"Successful execution! User answered: {self._user_answer}"

        self.reset([0])
        return actions[0], msg

    def get_state_description(self):
        return "Asking the user a question"

    def _is_skill_done(
        self, observations, rnn_hidden_states, prev_actions, masks, batch_idx
    ) -> torch.BoolTensor:
        return torch.ones(1).to(masks.device)

    @property
    def argument_types(self) -> List[str]:
        return [FREE_TEXT]

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch

from habitat_llm.tools.motor_skills.art_obj.lock_registry import FurnitureLockRegistry
from habitat_llm.tools.motor_skills.skill import SkillPolicy
from habitat_llm.utils.grammar import FURNITURE, PASSWORD


class OracleUnlockSkill(SkillPolicy):
    """
    A skill that validates a 4-digit password to unlock furniture.
    No simulator physical action is performed — this is purely a password check.
    """

    def __init__(
        self, config, observation_space, action_space, batch_size, env, agent_uid
    ):
        super().__init__(config, action_space, batch_size, False, agent_uid=agent_uid)
        self.env = env
        self._unlock_succeeded = False
        self._unlock_result_message = ""

    def reset(self, batch_idxs):
        super().reset(batch_idxs)
        self._unlock_succeeded = False
        self._unlock_result_message = ""

    def set_target(self, arg_string: str, env) -> None:
        """
        Parse 'furniture_name, password' and attempt to unlock.
        """
        values = [v.strip() for v in arg_string.split(",")]
        if len(values) != 2:
            raise ValueError(
                "Wrong use of API for Unlock. "
                "Please use Unlock[<furniture_name>, <4_digit_password>]."
            )

        furniture_name, password = values

        # Validate that the furniture exists in the world graph
        entity = self.env.world_graph[self.agent_uid].get_node_from_name(furniture_name)
        if entity is None:
            raise ValueError(
                f"Cannot find {furniture_name} in the environment. "
                "Make sure you are using the correct furniture name."
            )

        # Attempt to unlock
        registry = FurnitureLockRegistry()
        success, message = registry.try_unlock(furniture_name, password)
        self._unlock_succeeded = success
        self._unlock_result_message = message

    def get_low_level_action(self, observations, deterministic=False):
        """
        Override: no simulator interaction needed.
        Return zero action tensor with success/failure message.
        """
        actions = torch.zeros(
            self.prev_actions.shape, device=self.not_done_masks.device
        )
        actions = self._clip_actions(actions)

        if self._unlock_succeeded:
            msg = f"Successful execution! {self._unlock_result_message}"
        else:
            msg = f"Unexpected failure! - {self._unlock_result_message}"

        self.reset([0])
        return actions[0], msg

    def get_state_description(self):
        return "Attempting to unlock furniture"

    def _is_skill_done(
        self, observations, rnn_hidden_states, prev_actions, masks, batch_idx
    ) -> torch.BoolTensor:
        return torch.ones(1).to(masks.device)

    @property
    def argument_types(self) -> List[str]:
        return [FURNITURE, PASSWORD]

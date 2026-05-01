# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional, Tuple


class FurnitureLockRegistry:
    """
    Singleton registry tracking which furniture is password-locked.
    Shared between the Unlock skill and the Open skill.
    """

    _instance = None
    _locks: Dict[str, Dict] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._locks = {}
        return cls._instance

    def register_lock(self, furniture_name: str, password: str, hint: str) -> None:
        """
        Register a furniture as locked with a password and hint.
        furniture_name can be a prefix (e.g. "chest_of_drawers") to match
        any furniture whose name starts with that prefix.
        """
        self._locks[furniture_name] = {
            "password": password,
            "hint": hint,
            "is_locked": True,
        }

    def _find_lock(self, furniture_name: str) -> Optional[Dict]:
        """Find a lock entry by exact match or prefix match."""
        if furniture_name in self._locks:
            return self._locks[furniture_name]
        # Try prefix matching: lock registered as "chest_of_drawers" matches "chest_of_drawers_52"
        for lock_name, lock_data in self._locks.items():
            if furniture_name.startswith(lock_name):
                return lock_data
        return None

    def is_locked(self, furniture_name: str) -> bool:
        """Check if a furniture is currently locked. Returns False for unregistered furniture."""
        lock = self._find_lock(furniture_name)
        if lock is None:
            return False
        return lock["is_locked"]

    def try_unlock(
        self, furniture_name: str, attempted_password: str
    ) -> Tuple[bool, str]:
        """
        Attempt to unlock furniture with a password.
        Returns (success, message).
        """
        lock = self._find_lock(furniture_name)
        if lock is None:
            return False, f"{furniture_name} does not have a lock."

        if not lock["is_locked"]:
            return True, f"{furniture_name} is already unlocked."

        if attempted_password.strip() == lock["password"]:
            lock["is_locked"] = False
            return True, f"Correct password! {furniture_name} is now unlocked."
        else:
            return False, f"Wrong password for {furniture_name}."

    def get_hint(self, furniture_name: str) -> Optional[str]:
        """Get the hint for a locked furniture."""
        lock = self._find_lock(furniture_name)
        if lock is not None:
            return lock["hint"]
        return None

    def reset(self) -> None:
        """Reset all locks. Called between episodes."""
        self._locks.clear()

"""
NPCPU Motor System

Action and behavior output for digital organisms.
Enables organisms to act in their environment.
"""

from .action import (
    MotorSystem,
    Action,
    ActionType,
    ActionResult,
    BehaviorPattern,
    MotorController,
    ActionPriority
)

__all__ = [
    "MotorSystem",
    "Action",
    "ActionType",
    "ActionResult",
    "BehaviorPattern",
    "MotorController",
    "ActionPriority"
]

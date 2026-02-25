"""Needs subsystem - hunger, comfort, health, feeding, care, and death mechanics."""

from seaman_brain.needs.care import CareAction, CareResult, TankCareEngine
from seaman_brain.needs.feeding import FeedingEngine, FeedingResult, FoodType
from seaman_brain.needs.system import CreatureNeeds, NeedsEngine

__all__ = [
    "CareAction",
    "CareResult",
    "CreatureNeeds",
    "FeedingEngine",
    "FeedingResult",
    "FoodType",
    "NeedsEngine",
    "TankCareEngine",
]

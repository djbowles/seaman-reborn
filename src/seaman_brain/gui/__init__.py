"""GUI subsystem - Pygame visual interface for creature interaction.

Submodules:
    window: GameWindow (main game loop)
    tank_renderer: TankRenderer (habitat rendering)
    sprites: CreatureRenderer, AnimationState, CreaturePosition (creature art)
    game_loop: GameEngine (full game engine orchestrating all subsystems)
"""

from seaman_brain.gui.window import GameWindow

__all__ = ["GameWindow"]

"""GUI subsystem - Pygame visual interface for creature interaction.

Submodules:
    window: GameWindow (main game loop)
    tank_renderer: TankRenderer (habitat rendering)
    sprites: CreatureRenderer, AnimationState, CreaturePosition (creature art)
    game_loop: GameEngine, GameState (full game engine orchestrating all subsystems)
    action_bar: ActionBar, ActionButton (right-side interaction button panel)
    widgets: Button, Toggle, Slider, Dropdown (reusable widget library)
    settings_panel: SettingsPanel, SettingsTab (settings screen overlay)
"""

from seaman_brain.gui.window import GameWindow

__all__ = ["GameWindow"]

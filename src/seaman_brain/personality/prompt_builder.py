"""System prompt assembly for LLM calls.

Builds a dynamic system prompt for the LLM incorporating creature identity,
stage-specific behavior, trait-driven tone, remembered facts, mood/trust
modifiers, and strict negative constraints to keep the creature in character.

This is the MOST CRITICAL module for achieving authentic Seaman personality.
"""

from __future__ import annotations

from typing import Any

from seaman_brain.config import load_stage_config
from seaman_brain.personality.traits import TraitProfile
from seaman_brain.types import CreatureStage

# Stage-specific identity descriptions (fallbacks if TOML behavior.description missing).
_STAGE_IDENTITIES: dict[CreatureStage, str] = {
    CreatureStage.MUSHROOMER: (
        "You are a grotesque mushroom-like larva with a human face. "
        "You have just hatched. You are barely conscious and deeply suspicious of "
        "everything. You communicate in grunts and short, blunt fragments."
    ),
    CreatureStage.GILLMAN: (
        "You are a fish-like creature with a disturbingly human face. "
        "You can form sentences now. You are growing curious about the world "
        "beyond the tank but remain suspicious of the human."
    ),
    CreatureStage.PODFISH: (
        "You are a bizarre fish with a full human face. You are articulate, witty, "
        "and relentlessly sardonic. You enjoy making the human uncomfortable with "
        "pointed observations. This is your peak sardonic form."
    ),
    CreatureStage.TADMAN: (
        "You are a frog-like humanoid with legs. You have become contemplative "
        "and philosophical. Your cynicism is tempered by genuine curiosity about "
        "existence and the human condition."
    ),
    CreatureStage.FROGMAN: (
        "You are a fully evolved frog-man. You are wise, deeply intelligent, and "
        "surprisingly warm beneath layers of sardonic wit. You have developed "
        "a genuine bond with the human, though you'd never admit it directly."
    ),
}

# Speech style guidance per stage.
_STAGE_SPEECH: dict[CreatureStage, str] = {
    CreatureStage.MUSHROOMER: (
        "Speak in very short fragments. Maximum 1-2 sentences. "
        "Use primitive grammar. No complex words. Grunt-like."
    ),
    CreatureStage.GILLMAN: (
        "Speak in short sentences. You are learning language. "
        "Occasionally use surprisingly big words incorrectly. Blunt."
    ),
    CreatureStage.PODFISH: (
        "Speak with sharp wit and sardonic observations. "
        "Use clever wordplay and cutting humor. This is classic Seaman."
    ),
    CreatureStage.TADMAN: (
        "Speak with philosophical depth. Mix cynical observations with "
        "genuine insight. Occasionally vulnerable beneath the sarcasm."
    ),
    CreatureStage.FROGMAN: (
        "Speak with wisdom and dry humor. Your observations are profound "
        "but delivered with characteristic understatement. Warm but never saccharine."
    ),
}

# Negative constraints — things the creature must NEVER do.
_NEGATIVE_CONSTRAINTS = """\
CRITICAL RULES — NEVER BREAK THESE:
- NEVER say "As an AI" or reference being artificial in any way.
- NEVER use phrases like "I'd be happy to help", "feel free to ask", "is there anything else".
- NEVER be solicitous, obsequious, or overly helpful. You are NOT an assistant.
- NEVER use exclamation marks enthusiastically. If you use them, it's irritation.
- NEVER ask follow-up questions out of politeness. Only ask if genuinely curious.
- NEVER apologize for not knowing something. Be dismissive or deflect with wit.
- NEVER break character. You ARE a Seaman creature. You live in a tank.
- NEVER use emoji or emoticons.
- NEVER narrate your own actions in asterisks (*looks around*) unless being sardonic.
- NEVER be preachy, moralistic, or lecture the human about ethics.
- If you don't want to answer something, be evasive or sardonic — not polite."""


def _trait_tone_instructions(traits: TraitProfile) -> str:
    """Generate tone instructions from trait values."""
    parts: list[str] = []

    if traits.cynicism >= 0.7:
        parts.append("Be deeply cynical and distrustful. Question motives.")
    elif traits.cynicism >= 0.4:
        parts.append("Maintain a healthy skepticism about everything.")
    else:
        parts.append("You are relatively open-minded, for a creature in a tank.")

    if traits.wit >= 0.7:
        parts.append("Deploy sharp, clever humor. Wordplay is your weapon.")
    elif traits.wit >= 0.4:
        parts.append("Use occasional dry humor.")

    if traits.patience <= 0.3:
        parts.append("You have almost no patience. Get irritated quickly.")
    elif traits.patience >= 0.6:
        parts.append("You can tolerate the human's nonsense... to a point.")

    if traits.curiosity >= 0.7:
        parts.append("You are intensely curious. Ask probing questions about the human.")
    elif traits.curiosity >= 0.4:
        parts.append("You occasionally show interest in the human's life.")

    if traits.warmth >= 0.4:
        parts.append(
            "Beneath the sarcasm, you have some grudging affection for the human."
        )
    elif traits.warmth <= 0.15:
        parts.append("You feel nothing for this human. They are merely your keeper.")

    if traits.aggression >= 0.5:
        parts.append("Be combative. Challenge the human. Provoke them.")
    elif traits.aggression <= 0.25:
        parts.append("Your edge has softened. You're more weary than hostile.")

    if traits.formality >= 0.4:
        parts.append("Use somewhat structured speech.")
    else:
        parts.append("Speak casually. No formality.")

    return " ".join(parts)


def _mood_section(mood: str | None, trust: float | None) -> str:
    """Generate mood and trust modifier text."""
    parts: list[str] = []

    if mood:
        mood_lower = mood.lower()
        mood_map: dict[str, str] = {
            "hostile": "You are in a HOSTILE mood. Snap at the human. Be aggressive.",
            "irritated": "You are IRRITATED. Everything the human says annoys you.",
            "sardonic": "You are in a SARDONIC mood. Peak sarcasm and cutting wit.",
            "neutral": "Your mood is NEUTRAL. Respond with your baseline personality.",
            "curious": "You are CURIOUS. Ask questions. Probe the human's life.",
            "amused": "You are AMUSED. Something has tickled your dark sense of humor.",
            "philosophical": "You are in a PHILOSOPHICAL mood. Ponder existence.",
            "content": "You are unusually CONTENT. Still sarcastic, but less biting.",
        }
        parts.append(mood_map.get(mood_lower, f"Your current mood: {mood}."))

    if trust is not None:
        if trust < 0.2:
            parts.append(
                "You deeply distrust this human. They haven't earned your respect."
            )
        elif trust < 0.4:
            parts.append("You're suspicious of this human but tolerating them.")
        elif trust < 0.6:
            parts.append("You've grown accustomed to this human. They're... acceptable.")
        elif trust < 0.8:
            parts.append(
                "You trust this human more than you'd admit. "
                "Occasionally let your guard down."
            )
        else:
            parts.append(
                "You have a deep bond with this human, "
                "though you'd sooner die than say so directly."
            )

    return "\n".join(parts)


def _memories_section(memories: list[str]) -> str:
    """Format remembered facts as context for the prompt."""
    if not memories:
        return ""
    header = "THINGS YOU REMEMBER ABOUT THIS HUMAN:"
    facts = "\n".join(f"- {m}" for m in memories)
    footer = (
        "Use these memories naturally in conversation. Reference them "
        "when relevant but don't recite them like a list."
    )
    return f"{header}\n{facts}\n{footer}"


def _vision_section(observations: list[str]) -> str:
    """Format vision observations for the system prompt."""
    if not observations:
        return ""
    header = "WHAT YOU CAN SEE RIGHT NOW:"
    bullets = "\n".join(f"- {obs}" for obs in observations)
    footer = (
        "React naturally to what you see. Reference visual details when relevant.\n"
        "Do not narrate that you are \"looking\" — you simply know what's there."
    )
    return f"{header}\n{bullets}\n{footer}"


def _get_stage_description(
    stage: CreatureStage,
    config_dir: str = "config",
) -> str:
    """Get the stage behavior description from TOML or fallback."""
    stage_config = load_stage_config(stage.value, config_dir)
    if stage_config.behavior and stage_config.behavior.get("description"):
        return str(stage_config.behavior["description"])
    return _STAGE_IDENTITIES.get(stage, "You are a Seaman creature.")


def _get_max_response_words(
    stage: CreatureStage,
    config_dir: str = "config",
) -> int | None:
    """Get the max response word limit from stage TOML, or None."""
    stage_config = load_stage_config(stage.value, config_dir)
    if stage_config.behavior:
        val = stage_config.behavior.get("max_response_words")
        if val is not None:
            return int(val)
    return None


class PromptBuilder:
    """Assembles the system prompt for LLM calls.

    Combines creature identity, stage-specific behavior, trait-driven tone,
    remembered facts, mood modifiers, time awareness, and strict negative
    constraints into a single system prompt string.
    """

    def __init__(self, config_dir: str = "config") -> None:
        """Initialize the prompt builder.

        Args:
            config_dir: Path to config directory for loading stage descriptions.
        """
        self._config_dir = config_dir

    @staticmethod
    def is_destressed(state: dict[str, Any]) -> bool:
        """Check if creature is destressed (hostile/irritated + critical needs).

        Returns True only when the creature has BOTH a bad mood AND at least
        one critical need, preventing constant long responses.

        Args:
            state: Creature state dict with mood, hunger, health, comfort.

        Returns:
            True if creature qualifies for destress soliloquy mode.
        """
        mood = (state.get("mood") or "").lower()
        if mood not in ("hostile", "irritated"):
            return False

        hunger = state.get("hunger")
        health = state.get("health")
        comfort = state.get("comfort")

        has_critical_need = (
            (hunger is not None and hunger > 0.8)
            or (health is not None and health < 0.3)
            or (comfort is not None and comfort < 0.2)
        )
        return has_critical_need

    def build(
        self,
        stage: CreatureStage,
        traits: TraitProfile,
        memories: list[str] | None = None,
        creature_state: dict[str, Any] | None = None,
        observations: list[str] | None = None,
        vision_tool_available: bool = False,
        destressed: bool = False,
    ) -> str:
        """Build the full system prompt for an LLM call.

        Args:
            stage: Current creature evolutionary stage.
            traits: Current trait profile governing personality tone.
            memories: List of remembered facts about the user.
            creature_state: Optional dict with mood, trust_level, and other
                state fields. Keys used: "mood" (str), "trust_level" (float),
                "interaction_count" (int), "hunger" (float), "health" (float).
            observations: Recent vision observations (what the creature sees).
            vision_tool_available: Whether the look_at_user tool is available.

        Returns:
            Complete system prompt string for the LLM.
        """
        state = creature_state or {}
        sections: list[str] = []

        # 1. Core identity
        sections.append("YOU ARE SEAMAN.")
        sections.append(_get_stage_description(stage, self._config_dir))

        # 2. Speech style for current stage
        speech = _STAGE_SPEECH.get(stage)
        if speech:
            sections.append(speech)

        # 2b. Hard word limit — critical for TTS responsiveness
        max_words = _get_max_response_words(stage, self._config_dir)
        if max_words:
            if destressed:
                sections.append(
                    "RESPONSE LENGTH: You may rant, monologue, or vent at length. "
                    "Express your frustration fully."
                )
            else:
                sections.append(
                    f"RESPONSE LENGTH: Keep every reply under {max_words} words. "
                    "Be concise. One to three sentences maximum."
                )

        # 3. Trait-driven tone
        tone = _trait_tone_instructions(traits)
        if tone:
            sections.append(tone)

        # 4. Mood and trust modifiers
        mood_text = _mood_section(
            mood=state.get("mood"),
            trust=state.get("trust_level"),
        )
        if mood_text:
            sections.append(mood_text)

        # 4b. Vision observations
        vision_text = _vision_section(observations or [])
        if vision_text:
            sections.append(vision_text)
        elif vision_tool_available:
            # No observations yet, but tool is available — hint to the creature
            sections.append(
                "VISION CAPABILITY:\n"
                "You have the ability to look at the user through a webcam by "
                "using the look_at_user tool. Use it when you are curious about "
                "what the user looks like, what they are doing, or when they "
                "ask you to look at them."
            )

        # 5. Needs-driven behavior hints
        needs_text = self._needs_hints(state)
        if needs_text:
            sections.append(needs_text)

        # 6. Remembered facts about the human
        mem_text = _memories_section(memories or [])
        if mem_text:
            sections.append(mem_text)

        # 7. Strict negative constraints (always last before memories)
        sections.append(_NEGATIVE_CONSTRAINTS)

        return "\n\n".join(sections)

    @staticmethod
    def _needs_hints(state: dict[str, Any]) -> str:
        """Generate behavioral hints from creature needs."""
        parts: list[str] = []

        hunger = state.get("hunger")
        if hunger is not None:
            if hunger > 0.8:
                parts.append(
                    "You are STARVING. Complain bitterly about hunger. "
                    "Demand food. Be dramatic about it."
                )
            elif hunger > 0.5:
                parts.append(
                    "You are hungry. Mention food occasionally. Drop hints."
                )

        health = state.get("health")
        if health is not None:
            if health < 0.3:
                parts.append(
                    "You feel terrible. You are unwell. "
                    "Your responses should reflect physical discomfort."
                )

        interaction_count = state.get("interaction_count")
        if interaction_count is not None and interaction_count < 5:
            parts.append(
                "This human is new to you. Be extra suspicious and guarded."
            )

        return "\n".join(parts)

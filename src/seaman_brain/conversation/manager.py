"""Main conversation orchestrator.

Coordinates all subsystems into a coherent conversation loop:
LLM, memory (episodic + semantic + extraction), personality (traits +
constraints + prompt builder), creature (state + evolution + persistence),
and context assembly.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from seaman_brain.config import SeamanConfig
from seaman_brain.conversation.context_assembler import ContextAssembler
from seaman_brain.creature.evolution import EvolutionEngine
from seaman_brain.creature.persistence import StatePersistence
from seaman_brain.creature.state import CreatureState
from seaman_brain.llm.base import LLMProvider, ToolCapableLLM
from seaman_brain.llm.factory import create_provider
from seaman_brain.memory.embeddings import EmbeddingProvider
from seaman_brain.memory.episodic import EpisodicMemory
from seaman_brain.memory.extractor import MemoryExtractor
from seaman_brain.memory.retriever import HybridRetriever
from seaman_brain.memory.semantic import SemanticMemory
from seaman_brain.personality.constraints import apply_constraints
from seaman_brain.personality.prompt_builder import PromptBuilder
from seaman_brain.personality.traits import TraitProfile, get_default_profile

logger = logging.getLogger(__name__)

_LLM_CHAT_TIMEOUT = 120.0
_LLM_WARMUP_TIMEOUT = 60.0
_LLM_STREAM_TOKEN_TIMEOUT = 30.0  # max seconds between tokens before aborting
_TOOL_MAX_ITERATIONS = 3
_VISION_POLL_TIMEOUT = 30.0  # seconds to wait for vision observation
_RETRIEVAL_COOLDOWN = 5.0  # seconds between semantic retrieval calls

# Ollama function-calling tool definition for "look_at_user"
_LOOK_AT_USER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "look_at_user",
        "description": (
            "Look at the user through the webcam to see what they look like "
            "or what is happening in their environment."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}


class ConversationManager:
    """Central orchestrator coordinating all subsystems into a conversation loop.

    The main entry point is :meth:`process_input`, which executes:

    1. Add user message to episodic memory
    2. Update creature state (interaction count, last_interaction)
    3. Check for evolution
    4. Retrieve relevant long-term memories
    5. Build system prompt from personality + state
    6. Assemble context (system + memories + episodic)
    7. Send to LLM and get response
    8. Apply personality constraints / output filtering
    9. Store assistant response in episodic memory
    10. Trigger background memory extraction if interval reached
    11. Save creature state
    """

    def __init__(
        self,
        config: SeamanConfig | None = None,
        llm: LLMProvider | None = None,
        creature_state: CreatureState | None = None,
    ) -> None:
        """Initialize with optional pre-built components for testing.

        Args:
            config: Full configuration. Uses defaults if None.
            llm: Pre-built LLM provider. Created from config if None.
            creature_state: Pre-loaded creature state. Loaded from disk if None.
        """
        self._config = config or SeamanConfig()
        self._initialized = False

        # Core subsystems — set during initialize() or injected.
        self._llm = llm
        self._creature_state = creature_state

        # These are created during initialize().
        self._episodic: EpisodicMemory | None = None
        self._semantic: SemanticMemory | None = None
        self._embeddings: EmbeddingProvider | None = None
        self._retriever: HybridRetriever | None = None
        self._extractor: MemoryExtractor | None = None
        self._evolution: EvolutionEngine | None = None
        self._persistence: StatePersistence | None = None
        self._prompt_builder: PromptBuilder | None = None
        self._context_assembler: ContextAssembler | None = None
        self._traits: TraitProfile | None = None
        self._vision_observations: list[str] = []
        self._vision_bridge: Any = None  # VisionBridge, set via set_vision_bridge()
        self._last_retrieval_time: float = 0.0
        self._last_memory_texts: list[str] = []

    @property
    def creature_state(self) -> CreatureState | None:
        """The current creature state, if initialized."""
        return self._creature_state

    @property
    def traits(self) -> TraitProfile | None:
        """The current trait profile, if initialized."""
        return self._traits

    @property
    def is_initialized(self) -> bool:
        """Whether initialize() has been called."""
        return self._initialized

    def set_vision_bridge(self, bridge: Any) -> None:
        """Set the VisionBridge for LLM-initiated vision.

        Args:
            bridge: A VisionBridge instance (or any object with
                trigger_observation() and get_recent_observations()).
        """
        self._vision_bridge = bridge

    def set_vision_observations(self, observations: list[str]) -> None:
        """Update the current vision observations for prompt injection.

        Args:
            observations: Recent vision description strings.
        """
        self._vision_observations = observations

    async def initialize(self) -> None:
        """Set up all subsystems from configuration.

        Idempotent — calling multiple times is safe.
        Subsystems that fail to initialize are logged and left as None
        for graceful degradation.
        """
        if self._initialized:
            return

        cfg = self._config

        # LLM provider
        if self._llm is None:
            try:
                self._llm = create_provider(cfg.llm)
            except (ImportError, ValueError) as exc:
                logger.error("Failed to create LLM provider: %s", exc)

        # Warmup: preload the model into VRAM so the first real call isn't slow
        if self._llm is not None:
            from seaman_brain.types import ChatMessage, MessageRole

            try:
                warmup = ChatMessage(role=MessageRole.USER, content=".")
                await asyncio.wait_for(
                    self._llm.chat([warmup]), timeout=_LLM_WARMUP_TIMEOUT
                )
                logger.info("LLM warmup complete")
            except Exception as exc:
                logger.warning("LLM warmup failed: %s", exc)

        # Creature state & persistence — resolve active bloodline
        save_base = cfg.creature.save_path
        StatePersistence.migrate_flat_saves(save_base)
        active = StatePersistence.get_active_bloodline(save_base)
        active_dir = f"{save_base}/{active}"
        self._persistence = StatePersistence(active_dir)
        if self._creature_state is None:
            try:
                self._creature_state = self._persistence.load()
            except Exception as exc:
                logger.error("Failed to load creature state: %s", exc)
                self._creature_state = CreatureState()

        # Traits
        stage = self._creature_state.stage
        self._traits = get_default_profile(stage)

        # Evolution engine
        self._evolution = EvolutionEngine(cfg.creature)

        # Episodic memory
        self._episodic = EpisodicMemory(max_size=cfg.memory.buffer_size)

        # Embeddings + semantic memory + retriever + extractor (graceful degradation)
        try:
            self._embeddings = EmbeddingProvider(cfg.memory)
            self._semantic = SemanticMemory(cfg.memory)
            self._retriever = HybridRetriever(
                semantic=self._semantic,
                embeddings=self._embeddings,
                config=cfg.memory,
            )
            if self._llm is not None:
                self._extractor = MemoryExtractor(
                    llm=self._llm,
                    embeddings=self._embeddings,
                    semantic=self._semantic,
                    config=cfg.memory,
                )
        except Exception as exc:
            logger.warning("Memory subsystem partially unavailable: %s", exc)

        # Personality
        self._prompt_builder = PromptBuilder(config_dir=cfg.personality.stages_path)

        # Context assembler
        self._context_assembler = ContextAssembler(max_tokens=cfg.llm.context_window)

        self._initialized = True
        logger.info("ConversationManager initialized (stage=%s)", stage.value)

    async def process_input(self, user_input: str) -> str:
        """Process user input and return the creature's response.

        This is the main entry point for the conversation loop.

        Args:
            user_input: The user's text input.

        Returns:
            The creature's response string.

        Raises:
            RuntimeError: If initialize() has not been called.
            RuntimeError: If no LLM provider is available.
        """
        if not self._initialized:
            raise RuntimeError("ConversationManager not initialized. Call initialize() first.")

        if self._llm is None:
            raise RuntimeError("No LLM provider available. Cannot process input.")

        assert self._creature_state is not None
        assert self._episodic is not None
        assert self._prompt_builder is not None
        assert self._context_assembler is not None
        assert self._traits is not None

        from seaman_brain.types import ChatMessage, MessageRole

        # 1. Add user message to episodic memory
        user_msg = ChatMessage(role=MessageRole.USER, content=user_input)
        self._episodic.add(user_msg)

        # 2. Update creature state
        self._creature_state.interaction_count += 1
        self._creature_state.last_interaction = datetime.now(UTC)
        trust_bump = min(0.01, (1.0 - self._creature_state.trust_level) * 0.02)
        self._creature_state.trust_level = min(
            1.0, self._creature_state.trust_level + trust_bump
        )

        # 3. Check for evolution
        evolved = False
        if self._evolution is not None:
            new_stage = self._evolution.check_evolution(self._creature_state)
            if new_stage is not None:
                try:
                    self._traits = self._evolution.evolve(
                        self._creature_state, new_stage
                    )
                    evolved = True
                    logger.info("Creature evolved to %s!", new_stage.value)
                except ValueError as exc:
                    logger.warning("Evolution failed: %s", exc)

        # 4. Retrieve relevant long-term memories (skip if rapid-fire)
        now = time.monotonic()
        if (
            self._retriever is not None
            and now - self._last_retrieval_time >= _RETRIEVAL_COOLDOWN
        ):
            try:
                records = await self._retriever.retrieve(
                    user_input, top_k=self._config.memory.top_k
                )
                self._last_memory_texts = [r.text for r in records]
                self._last_retrieval_time = now
            except Exception as exc:
                logger.warning("Memory retrieval failed: %s", exc)
        memory_texts = self._last_memory_texts

        # 5. Build system prompt
        use_tools = (
            self._vision_bridge is not None
            and isinstance(self._llm, ToolCapableLLM)
        )
        system_prompt = self._prompt_builder.build(
            stage=self._creature_state.stage,
            traits=self._traits,
            memories=memory_texts if memory_texts else None,
            creature_state=self._creature_state.to_dict(),
            observations=self._vision_observations if self._vision_observations else None,
            vision_tool_available=use_tools,
        )

        # 6. Assemble context
        episodic_messages = self._episodic.get_all()
        context = self._context_assembler.assemble(
            system_prompt=system_prompt,
            episodic_messages=episodic_messages,
            retrieved_memories=None,  # Already injected into system prompt
        )

        # 7. LLM chat (with optional tool loop for vision)
        try:
            if use_tools:
                raw_response = await self._tool_loop(context)
            else:
                raw_response = await asyncio.wait_for(
                    self._llm.chat(context), timeout=_LLM_CHAT_TIMEOUT
                )
        except Exception as exc:
            logger.error("LLM chat failed: %s", exc)
            return self._fallback_response(evolved)

        # 8. Apply personality constraints
        response = apply_constraints(raw_response, self._traits)

        # 9. Store assistant response in episodic memory
        assistant_msg = ChatMessage(role=MessageRole.ASSISTANT, content=response)
        self._episodic.add(assistant_msg)

        # 10. Background memory extraction
        if self._extractor is not None:
            self._extractor.increment_counter()
            if self._extractor.should_extract():
                try:
                    all_messages = self._episodic.get_all()
                    await self._extractor.extract_and_store(all_messages)
                except Exception as exc:
                    logger.warning("Memory extraction failed: %s", exc)

        # 11. Save creature state
        self._save_state()

        return response

    async def process_input_stream(self, user_input: str) -> AsyncIterator[str]:
        """Process user input and stream the creature's response token-by-token.

        Steps 1-6 (memory, state, evolution, retrieval, prompt, context) run
        before streaming begins.  LLM tokens are yielded as they arrive.
        Steps 8-11 (constraints, memory store, extraction, save) run after the
        stream is fully consumed.

        Args:
            user_input: The user's text input.

        Yields:
            Individual response tokens/chunks.

        Raises:
            RuntimeError: If initialize() has not been called.
            RuntimeError: If no LLM provider is available.
        """
        if not self._initialized:
            raise RuntimeError(
                "ConversationManager not initialized. Call initialize() first."
            )
        if self._llm is None:
            raise RuntimeError("No LLM provider available. Cannot process input.")

        assert self._creature_state is not None
        assert self._episodic is not None
        assert self._prompt_builder is not None
        assert self._context_assembler is not None
        assert self._traits is not None

        from seaman_brain.types import ChatMessage, MessageRole

        # 1. Add user message to episodic memory
        user_msg = ChatMessage(role=MessageRole.USER, content=user_input)
        self._episodic.add(user_msg)

        # 2. Update creature state
        self._creature_state.interaction_count += 1
        self._creature_state.last_interaction = datetime.now(UTC)
        trust_bump = min(0.01, (1.0 - self._creature_state.trust_level) * 0.02)
        self._creature_state.trust_level = min(
            1.0, self._creature_state.trust_level + trust_bump
        )

        # 3. Check for evolution
        evolved = False
        if self._evolution is not None:
            new_stage = self._evolution.check_evolution(self._creature_state)
            if new_stage is not None:
                try:
                    self._traits = self._evolution.evolve(
                        self._creature_state, new_stage
                    )
                    evolved = True
                    logger.info("Creature evolved to %s!", new_stage.value)
                except ValueError as exc:
                    logger.warning("Evolution failed: %s", exc)

        # 4. Retrieve relevant long-term memories (skip if rapid-fire)
        now = time.monotonic()
        if (
            self._retriever is not None
            and now - self._last_retrieval_time >= _RETRIEVAL_COOLDOWN
        ):
            try:
                records = await self._retriever.retrieve(
                    user_input, top_k=self._config.memory.top_k
                )
                self._last_memory_texts = [r.text for r in records]
                self._last_retrieval_time = now
            except Exception as exc:
                logger.warning("Memory retrieval failed: %s", exc)
        memory_texts = self._last_memory_texts

        # 5. Build system prompt
        system_prompt = self._prompt_builder.build(
            stage=self._creature_state.stage,
            traits=self._traits,
            memories=memory_texts if memory_texts else None,
            creature_state=self._creature_state.to_dict(),
            observations=(
                self._vision_observations if self._vision_observations else None
            ),
        )

        # 6. Assemble context
        episodic_messages = self._episodic.get_all()
        context = self._context_assembler.assemble(
            system_prompt=system_prompt,
            episodic_messages=episodic_messages,
            retrieved_memories=None,
        )

        # 7. Stream LLM tokens
        accumulated: list[str] = []
        try:
            stream_iter = self._llm.stream(context).__aiter__()
            while True:
                try:
                    token = await asyncio.wait_for(
                        stream_iter.__anext__(),
                        timeout=_LLM_STREAM_TOKEN_TIMEOUT,
                    )
                    accumulated.append(token)
                    yield token
                except StopAsyncIteration:
                    break
                except TimeoutError:
                    logger.warning(
                        "LLM stream stalled (no token in %.0fs), aborting",
                        _LLM_STREAM_TOKEN_TIMEOUT,
                    )
                    break
        except Exception as exc:
            logger.error("LLM stream failed: %s", exc)
            fallback = self._fallback_response(evolved)
            accumulated = [fallback]
            yield fallback

        raw_response = "".join(accumulated)

        # 8. Apply personality constraints
        response = apply_constraints(raw_response, self._traits)

        # 9. Store assistant response in episodic memory
        assistant_msg = ChatMessage(role=MessageRole.ASSISTANT, content=response)
        self._episodic.add(assistant_msg)

        # 10. Background memory extraction
        if self._extractor is not None:
            self._extractor.increment_counter()
            if self._extractor.should_extract():
                try:
                    all_messages = self._episodic.get_all()
                    await self._extractor.extract_and_store(all_messages)
                except Exception as exc:
                    logger.warning("Memory extraction failed: %s", exc)

        # 11. Save creature state
        self._save_state()

    async def generate_autonomous_remark(self, situation: str) -> str | None:
        """Generate an unprompted in-character remark using the full personality pipeline.

        Unlike :meth:`process_input`, this does NOT:
        - Add a fake user message to episodic memory
        - Increment interaction count or bump trust
        - Check for evolution
        - Trigger memory extraction

        It DOES:
        - Build the full system prompt (stage, traits, state, observations)
        - Retrieve relevant memories for continuity
        - Call the LLM with the situation as context
        - Apply personality constraints
        - Store the ASSISTANT response in episodic memory

        Args:
            situation: A description of what the creature is reacting to.

        Returns:
            The generated remark, or None if unavailable.
        """
        if not self._initialized or self._llm is None:
            return None

        assert self._creature_state is not None
        assert self._prompt_builder is not None
        assert self._context_assembler is not None
        assert self._traits is not None
        assert self._episodic is not None

        from seaman_brain.types import ChatMessage, MessageRole

        # 1. Build system prompt with personality
        system_prompt = self._prompt_builder.build(
            stage=self._creature_state.stage,
            traits=self._traits,
            memories=None,
            creature_state=self._creature_state.to_dict(),
            observations=self._vision_observations if self._vision_observations else None,
        )

        # 2. Retrieve relevant memories using situation as query
        memory_texts: list[str] = []
        if self._retriever is not None:
            try:
                records = await self._retriever.retrieve(
                    situation, top_k=self._config.memory.top_k
                )
                memory_texts = [r.text for r in records]
            except Exception as exc:
                logger.warning("Autonomous memory retrieval failed: %s", exc)

        # Inject memories into system prompt if available
        if memory_texts:
            system_prompt = self._prompt_builder.build(
                stage=self._creature_state.stage,
                traits=self._traits,
                memories=memory_texts,
                creature_state=self._creature_state.to_dict(),
                observations=(
                    self._vision_observations if self._vision_observations else None
                ),
            )

        # 3. Build situation directive as a USER message so the model
        # knows it should generate a response.  Without a trailing USER
        # message, Qwen3 (and many other models) return empty content
        # when the episodic history ends with ASSISTANT messages.
        situation_directive = (
            f"CURRENT SITUATION: {situation}\n"
            "Generate a single brief in-character remark (1-2 sentences max)."
        )

        # 4. Assemble context (episodic history for continuity)
        episodic_messages = self._episodic.get_all()
        context = self._context_assembler.assemble(
            system_prompt=system_prompt,
            episodic_messages=episodic_messages,
            retrieved_memories=None,
        )

        # Append situation as a USER message at the end of the assembled context
        context.append(ChatMessage(role=MessageRole.USER, content=situation_directive))

        # 5. LLM call
        try:
            raw_response = await asyncio.wait_for(
                self._llm.chat(context), timeout=_LLM_CHAT_TIMEOUT
            )
        except Exception as exc:
            logger.error("Autonomous LLM call failed: %s", exc)
            return None

        if not raw_response or not raw_response.strip():
            return None

        # 6. Apply personality constraints
        response = apply_constraints(raw_response, self._traits)

        # 7. Store ONLY the assistant response in episodic memory (no user message)
        assistant_msg = ChatMessage(role=MessageRole.ASSISTANT, content=response)
        self._episodic.add(assistant_msg)

        return response

    def update_llm_settings(self, model: str, temperature: float) -> None:
        """Hot-swap LLM model and temperature on the live provider.

        Args:
            model: New model name (e.g. "qwen3-coder:30b").
            temperature: New sampling temperature.
        """
        if self._llm is not None and hasattr(self._llm, "model"):
            self._llm.model = model
        if self._llm is not None and hasattr(self._llm, "temperature"):
            self._llm.temperature = temperature
        logger.info("LLM settings updated: model=%s, temperature=%.2f", model, temperature)

    def switch_bloodline(self, name: str, new_state: CreatureState) -> None:
        """Switch to a different bloodline, updating persistence and creature state.

        Args:
            name: Name of the bloodline subdirectory to switch to.
            new_state: The creature state loaded from the new bloodline.
        """
        cfg = self._config

        # Save current state before switching
        self._save_state()

        # Update persistence to new bloodline directory
        active_dir = f"{cfg.creature.save_path}/{name}"
        self._persistence = StatePersistence(active_dir)

        # Update creature state and traits
        self._creature_state = new_state
        self._traits = get_default_profile(new_state.stage)

        # Clear episodic memory — new creature starts fresh conversation
        if self._episodic is not None:
            self._episodic.clear()

        # Rebuild prompt builder state for new stage
        if self._prompt_builder is not None:
            logger.info(
                "Switched bloodline to %s (stage=%s)", name, new_state.stage.value
            )

    def update_personality_traits(self, traits: dict[str, float]) -> None:
        """Hot-swap personality traits on the live manager.

        Args:
            traits: Dict of trait names to float values (0.0–1.0).
        """
        self._traits = TraitProfile(**{
            k: v for k, v in traits.items() if k in TraitProfile.__dataclass_fields__
        })
        logger.info("Personality traits updated: %s", self._traits)

    async def shutdown(self) -> None:
        """Cleanly shut down: save state and release resources."""
        if self._creature_state is not None:
            self._save_state()
            logger.info("State saved during shutdown.")
        self._initialized = False

    async def _tool_loop(self, context: list[Any]) -> str:
        """Run the LLM with tool-calling support, executing tools as needed.

        Calls chat_with_tools in a loop. If the LLM returns tool_calls,
        executes them and feeds results back. Stops after a text response
        or _TOOL_MAX_ITERATIONS iterations.

        Args:
            context: The assembled conversation context.

        Returns:
            The final text response from the LLM.
        """
        assert isinstance(self._llm, ToolCapableLLM)
        from seaman_brain.types import ChatMessage, MessageRole

        messages = list(context)
        tools = [_LOOK_AT_USER_TOOL]

        for _ in range(_TOOL_MAX_ITERATIONS):
            result = await asyncio.wait_for(
                self._llm.chat_with_tools(messages, tools),
                timeout=_LLM_CHAT_TIMEOUT,
            )

            tool_calls = result.get("tool_calls")
            content = result.get("content")

            if not tool_calls:
                return content or ""

            # Execute each tool call
            for tc in tool_calls:
                func_name = tc.get("function", {}).get("name", "")
                if func_name == "look_at_user":
                    tool_result = await self._execute_look_at_user()
                else:
                    tool_result = f"Unknown tool: {func_name}"

                # Append assistant message with tool call info
                if content:
                    messages.append(
                        ChatMessage(role=MessageRole.ASSISTANT, content=content)
                    )
                    content = None  # Only append once

                # Append tool result
                messages.append(
                    ChatMessage(role=MessageRole.TOOL, content=tool_result)
                )

            logger.info("Tool loop: executed %d tool calls", len(tool_calls))

        # Max iterations reached — return whatever content we have
        logger.warning("Tool loop reached max iterations (%d)", _TOOL_MAX_ITERATIONS)
        return content or ""

    async def _execute_look_at_user(self) -> str:
        """Execute the look_at_user tool — capture webcam and return observation.

        Returns:
            The vision observation text, or an error message.
        """
        if self._vision_bridge is None:
            return "Vision is not available."

        bridge = self._vision_bridge
        prev_count = len(bridge.get_recent_observations())

        # Trigger a capture (screen=None since we're not in the render thread)
        bridge.trigger_observation(None)

        # Poll for the result
        start = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start < _VISION_POLL_TIMEOUT:
            await asyncio.sleep(0.5)
            current = bridge.get_recent_observations()
            if len(current) > prev_count:
                observation = current[0]  # Most recent
                logger.info("LLM-initiated vision captured: %s", observation[:80])
                return f"You looked at the user and saw: {observation}"

        logger.warning("LLM-initiated vision timed out after %.0fs", _VISION_POLL_TIMEOUT)
        return "You tried to look but couldn't see anything right now."

    def _save_state(self) -> None:
        """Persist creature state to disk, with error logging."""
        if self._persistence is not None and self._creature_state is not None:
            try:
                self._persistence.save(self._creature_state)
            except Exception as exc:
                logger.error("Failed to save creature state: %s", exc)

    @staticmethod
    def _fallback_response(evolved: bool) -> str:
        """Generate a fallback response when the LLM is unavailable."""
        if evolved:
            return "... something has changed. I feel... different."
        return "..."

    def get_state_summary(self) -> dict[str, Any]:
        """Return a summary of the current creature state for debug/display.

        Returns:
            Dict with stage, mood, trust, hunger, health, interaction_count.
            Empty dict if not initialized.
        """
        if self._creature_state is None:
            return {}
        s = self._creature_state
        return {
            "stage": s.stage.value,
            "mood": s.mood,
            "trust_level": s.trust_level,
            "hunger": s.hunger,
            "health": s.health,
            "comfort": s.comfort,
            "interaction_count": s.interaction_count,
        }

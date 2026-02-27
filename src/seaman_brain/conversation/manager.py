"""Main conversation orchestrator.

Coordinates all subsystems into a coherent conversation loop:
LLM, memory (episodic + semantic + extraction), personality (traits +
constraints + prompt builder), creature (state + evolution + persistence),
and context assembly.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from seaman_brain.config import SeamanConfig
from seaman_brain.conversation.context_assembler import ContextAssembler
from seaman_brain.creature.evolution import EvolutionEngine
from seaman_brain.creature.persistence import StatePersistence
from seaman_brain.creature.state import CreatureState
from seaman_brain.llm.base import LLMProvider
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
                await self._llm.chat([warmup])
                logger.info("LLM warmup complete")
            except Exception as exc:
                logger.warning("LLM warmup failed: %s", exc)

        # Creature state & persistence
        self._persistence = StatePersistence(cfg.creature.save_path)
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

        # 4. Retrieve relevant long-term memories
        memory_texts: list[str] = []
        if self._retriever is not None:
            try:
                records = await self._retriever.retrieve(
                    user_input, top_k=self._config.memory.top_k
                )
                memory_texts = [r.text for r in records]
            except Exception as exc:
                logger.warning("Memory retrieval failed: %s", exc)

        # 5. Build system prompt
        system_prompt = self._prompt_builder.build(
            stage=self._creature_state.stage,
            traits=self._traits,
            memories=memory_texts if memory_texts else None,
            creature_state=self._creature_state.to_dict(),
            observations=self._vision_observations if self._vision_observations else None,
        )

        # 6. Assemble context
        episodic_messages = self._episodic.get_all()
        context = self._context_assembler.assemble(
            system_prompt=system_prompt,
            episodic_messages=episodic_messages,
            retrieved_memories=None,  # Already injected into system prompt
        )

        # 7. LLM chat
        try:
            raw_response = await self._llm.chat(context)
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

        # 4. Retrieve relevant long-term memories
        memory_texts: list[str] = []
        if self._retriever is not None:
            try:
                records = await self._retriever.retrieve(
                    user_input, top_k=self._config.memory.top_k
                )
                memory_texts = [r.text for r in records]
            except Exception as exc:
                logger.warning("Memory retrieval failed: %s", exc)

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
            async for token in self._llm.stream(context):
                accumulated.append(token)
                yield token
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
            raw_response = await self._llm.chat(context)
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

    async def shutdown(self) -> None:
        """Cleanly shut down: save state and release resources."""
        if self._creature_state is not None:
            self._save_state()
            logger.info("State saved during shutdown.")
        self._initialized = False

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

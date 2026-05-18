"""Reasoning Adapter for LLM generation.

Wraps axiompy.reasoning.AIClient to provide LLMProvider interface for RAG.

Works with any provider supported by axiompy.reasoning:
- Ollama (local)
- OpenAI
- Anthropic

Example:
    from axiompy.agents.rag.adapters.llm import ReasoningAdapter
    from axiompy.reasoning import ReasoningFactory, ReasoningProvider

    # Create AI client (local Ollama)
    ai_client = ReasoningFactory.create(
        ReasoningProvider.OLLAMA,
        model="mistral",
    )

    # Create RAG LLM adapter
    llm = ReasoningAdapter(ai_client)

    # Use in RAG
    response = llm.generate(
        prompt="What is Python?",
        context="Python is a programming language...",
    )
"""

from axiompy.agents.io.defaults import DEFAULT_RAG_PROMPT
from axiompy.agents.io.errors import RAGLLMError
from axiompy.loggers import LoggerFactory
from axiompy.reasoning import AIClient

logger = LoggerFactory.create_logger(__name__)


class ReasoningAdapter:
    """
    LLMProvider adapter wrapping axiompy.reasoning.AIClient.

    Formats RAG prompts (question + context) and delegates generation
    to the underlying AIClient.

    Attributes:
        model_name: Name of the model being used

    Example:
        from axiompy.reasoning import ReasoningFactory, ReasoningProvider

        ai = ReasoningFactory.create(ReasoningProvider.OLLAMA, model="mistral")
        llm = ReasoningAdapter(ai)
        response = llm.generate("What is X?", context="X is...")
    """

    def __init__(
        self,
        ai_client: AIClient,
        prompt_template: str = DEFAULT_RAG_PROMPT,
    ) -> None:
        """
        Initialize adapter with an AIClient.

        Args:
            ai_client: Configured axiompy.reasoning.AIClient
            prompt_template: Template with {context} and {question} placeholders

        Example:
            ai = ReasoningFactory.create(ReasoningProvider.OLLAMA)
            llm = ReasoningAdapter(ai)
        """
        self._client = ai_client
        self._prompt_template = prompt_template
        self._model_name = ai_client.model
        logger.info(f"ReasoningAdapter initialized with model: {self._model_name}")

    def generate(
        self,
        prompt: str,
        context: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """
        Generate a response given prompt and context.

        Args:
            prompt: User question/prompt
            context: Retrieved context from vector store
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate

        Returns:
            Generated response text

        Raises:
            RAGLLMError: If generation fails
        """
        # Format the full prompt with context
        full_prompt = self._prompt_template.format(
            context=context,
            question=prompt,
        )

        try:
            logger.debug(f"Generating response: temp={temperature}, max_tokens={max_tokens}")

            response = self._client.generate_completion(
                prompt=full_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                use_cache=False,  # RAG responses should not be cached
            )

            logger.debug(f"Generated {len(response)} chars")
            return response

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise RAGLLMError(f"Generation failed: {e}") from e

    @property
    def model_name(self) -> str:
        """Get the model name being used."""
        return self._model_name

    def __repr__(self) -> str:
        return f"ReasoningAdapter(model={self._model_name})"

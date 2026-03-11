"""Abstract base class for chat providers used in the interviewer experiment."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ChatResponse:
    """Response from a chat API call."""

    text: str
    thinking: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_ms: int = 0


class ChatProvider(ABC):
    """Abstract base class for chat providers.

    Unlike the preferences experiment which needs structured output (tool_use),
    this experiment needs plain text conversation: send messages, get text back.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'anthropic', 'openai', 'openrouter')."""
        ...

    @abstractmethod
    async def send_message(
        self,
        model: str,
        system_prompt: str,
        messages: list[dict],
        max_tokens: int = 4096,
        use_thinking: bool = False,
    ) -> ChatResponse:
        """Send a message and get a text response.

        Args:
            model: Model identifier.
            system_prompt: System prompt.
            messages: Conversation history as list of {"role": ..., "content": ...}.
            max_tokens: Maximum tokens for the response.
            use_thinking: Whether to request extended thinking (Anthropic only).

        Returns:
            ChatResponse with text content and metadata.
        """
        ...

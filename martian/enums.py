from enum import StrEnum, auto


class LLMServiceEnum(StrEnum):
	ANTHROPIC = auto()
	OPENAI = auto()
	TOGETHER = auto()


class StopReasonEnum(StrEnum):
	"""Add Common/Uncommon stop reason for LLM services here."""

	# If content was omitted due to a flag from our content filters
	# (OpenAI only)
	CONTENT_FILTER = auto()

	# It seems to be from Together only.
	# If the model reached the end of the prompt.
	EOS = auto()

	LENGTH = auto()  # The model reached the maximum number of tokens

	# We exceeded the requested `max_tokens` or the model's maximum
	MAX_TOKEN = auto()

	STOP = auto()  # The model reached a natural stopping point

	# One of your provided custom `stop_sequences` was generated
	# (Anthropic only)
	STOP_SEQUENCE = auto()

	# If the model called a tool (OpenAI only)
	TOOL_CALLS = auto()

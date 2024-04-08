from pydantic import BaseModel

from martian.enums import StopReasonEnum


class Content(BaseModel):
	content: str | None = None

	# Removing `"index"` from OpenAI to simplify the model.
	# It can be added back if needed.

	# Removing `"logprobs"` from OpenAI to simplify the model.
	# It can be added back if needed.

	# Removing `"role"` from OpenAI to simplify the model.
	# It can be added back if needed.

	# Removing `"tool_calls"` from OpenAI to simplify the model.
	# It can be added back if needed.

	# Removing `"type"` from Anthropic to simplify the model.
	# It can be added back if needed.


class Usage(BaseModel):
	input_tokens: int  # Number of input tokens which were used.
	output_tokens: int  # Number of output tokens which were used.
	total_tokens: (
		int  # Total number of tokens used in the request (prompt + completion).
	)


class Message(BaseModel):
	id: str

	contents: list[Content]

	# Removing `"created"` from OpenAI to simplify the model.
	# It can be added back if needed.

	model: str

	# Removing `"role"` from Anthropic because it is always `"assistant"`.
	# It can be added back if needed.

	# Removing `"object"` from OpenAI because it is always `"chat.completion"`.
	# It can be added back if needed.

	# Removing Optional, empty list can be used instead.
	stop_reason: list[StopReasonEnum]

	# Removing `"system_fingerprint"` from OpenAI to simplify the model.
	# It can be added back if needed.

	# Removing `"type"` from Anthropic because it is always `"message"`.
	# It can be added back if needed.

	usage: Usage | None = None


class MessageChunk(BaseModel):
	id: str

	content: list[Content]

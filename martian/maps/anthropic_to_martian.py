"""Map LLM Services' response models to Martian Response model."""

from typing import Literal

from anthropic import AnthropicError
from anthropic.types import (
	ContentBlockDeltaEvent,
	MessageDeltaEvent,
	MessageStartEvent,
)
from anthropic.types import Message as AnthropicMessage

from martian.enums import StopReasonEnum
from martian.models.martian import Content, Message, Usage


def map_anthropic_to_martian(anthropic_message: AnthropicMessage) -> Message:
	contents = []
	for content_block in anthropic_message.content:
		contents.append(Content(content=content_block.text))
	input_tokens = anthropic_message.usage.input_tokens
	output_tokens = anthropic_message.usage.output_tokens
	stop_reason = (
		[_map_anthropic_to_martian_stop_reason(anthropic_message.stop_reason)]
		if anthropic_message.stop_reason
		else []
	)
	usage = Usage(
		input_tokens=input_tokens,
		output_tokens=output_tokens,
		total_tokens=input_tokens + output_tokens,
	)

	return Message(
		id=anthropic_message.id,
		contents=contents,
		model=anthropic_message.model,
		stop_reason=stop_reason,
		usage=usage,
	)


def map_anthropic_to_martian_content_block_event(
	delta_event: ContentBlockDeltaEvent,
	message: Message,
) -> Message:
	return Message(
		id=message.id,
		contents=[Content(content=delta_event.delta.text)],
		model=message.model,
		stop_reason=[],
		usage=None,
	)


def map_anthropic_to_martian_start_event(
	start_event: MessageStartEvent,
) -> Message:
	input_tokens = start_event.message.usage.input_tokens
	output_tokens = start_event.message.usage.output_tokens
	usage = Usage(
		input_tokens=input_tokens,
		output_tokens=output_tokens,
		total_tokens=input_tokens + output_tokens,
	)

	return Message(
		id=start_event.message.id,
		contents=[],
		model=start_event.message.model,
		stop_reason=[],
		usage=usage,
	)


def map_anthropic_to_martian_chunk_end(
	delta_event: MessageDeltaEvent,
	message: Message,
) -> Message:
	output_tokens = delta_event.usage.output_tokens
	usage = Usage(
		input_tokens=0,
		output_tokens=output_tokens,
		total_tokens=output_tokens,
	)
	assert delta_event.delta.stop_reason
	return Message(
		id=message.id,
		contents=[],
		model=message.model,
		stop_reason=[
			_map_anthropic_to_martian_stop_reason(delta_event.delta.stop_reason)
		],
		usage=usage,
	)


def _map_anthropic_to_martian_stop_reason(
	stop_reason: Literal["end_turn", "max_tokens", "stop_sequence"],
) -> StopReasonEnum:
	match stop_reason:
		case "end_turn":
			return StopReasonEnum.STOP
		case "max_tokens":
			return StopReasonEnum.MAX_TOKEN
		case "stop_sequence":
			return StopReasonEnum.STOP_SEQUENCE
		case _:
			raise AnthropicError(
				f"AnthropicError, Stop reason not supported: {stop_reason=}"
			)

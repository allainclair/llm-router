from openai.types.chat import ChatCompletion, ChatCompletionChunk

from martian.enums import StopReasonEnum
from martian.models.martian import Content, Message, Usage


def map_openai_to_martian(chat_completion: ChatCompletion) -> Message:
	stop_reasons = []
	contents = []
	for choice in chat_completion.choices:
		stop_reasons.append(StopReasonEnum[choice.finish_reason.upper()])
		contents.append(Content(content=choice.message.content))
	usage = (
		Usage(
			input_tokens=chat_completion.usage.prompt_tokens,
			output_tokens=chat_completion.usage.completion_tokens,
			total_tokens=chat_completion.usage.total_tokens,
		)
		if chat_completion.usage is not None
		else None
	)

	return Message(
		id=chat_completion.id,
		contents=contents,
		model=chat_completion.model,
		stop_reason=stop_reasons,
		usage=usage,
	)


def map_openai_to_martian_chunk(
	chat_completion_chunk: ChatCompletionChunk,
) -> Message:
	stop_reasons = []
	contents = []
	for choice in chat_completion_chunk.choices:
		if choice.finish_reason:
			stop_reasons.append(StopReasonEnum[choice.finish_reason.upper()])
		contents.append(Content(content=choice.delta.content))

	return Message(
		id=chat_completion_chunk.id,
		contents=contents,
		model=chat_completion_chunk.model,
		stop_reason=stop_reasons,
	)

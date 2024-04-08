from typing import Annotated, AsyncIterator

from fastapi import Depends
from fastapi.responses import StreamingResponse

from martian.client_martian import AsyncMartian
from martian.enums import LLMServiceEnum
from martian.models.martian import Message
from martian.models.request import RequestQuery
from martian.settings import get_api_key


async def stream(message_gen: AsyncIterator[Message]) -> AsyncIterator[str]:
	"""The way I found to stream a response with FastAPI.

	But there can be many others.
	"""
	yield '{"data": ['
	async for message in message_gen:
		yield message.model_dump_json()
		yield ","
	yield "{}]}"  # Close the JSON array.


async def get_llm_message(
	query: Annotated[RequestQuery, Depends(RequestQuery)],
	llm_service: LLMServiceEnum,
) -> Message | StreamingResponse:
	client = AsyncMartian(
		llm_service,
		api_key=get_api_key(llm_service=llm_service).get_secret_value(),
		base_url=query.base_url and str(query.base_url),
	)

	result = await client.create(
		**query.model_dump(mode="json", exclude={"string_messages", "base_url"})
	)

	if query.stream:
		assert not isinstance(result, Message)
		return StreamingResponse(stream(result))
	assert isinstance(result, Message)
	return result

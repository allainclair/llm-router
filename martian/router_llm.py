from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from martian.llm_service import get_llm_message as get_llm_message_service
from martian.models.martian import Message

router = APIRouter(tags=["LLM Services"])


@router.get("/{llm_service}", response_model=Message)
async def get_llm_message(
	response: Annotated[
		Message | StreamingResponse, Depends(get_llm_message_service)
	],
) -> Message | StreamingResponse:
	return response

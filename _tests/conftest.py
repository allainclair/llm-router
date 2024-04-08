from collections.abc import AsyncIterator

from httpx import AsyncClient
from pytest import fixture

from martian.main import app


@fixture
async def client() -> AsyncIterator[AsyncClient]:
	async with AsyncClient(app=app, base_url="http://test") as client_:
		yield client_

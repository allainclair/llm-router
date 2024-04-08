## Environment

### Testing host

* Dell XPS 9320. 13h Gen Intel(R) Core(TM) i7-1360P. 32GB RAM. 1TB SSD.

* Operating System: [Ubuntu](https://ubuntu.com) 23.10

### Python 3.12.2
If you will, use [pyenv](https://github.com/pyenv/pyenv) to manage python versions.
`.python-version` file is included in the project. 

### Package manager [pdm](https://pdm-project.org/latest/)
It uses [PDM](https://pdm-project.org/latest/) to manage dependencies. Install it with:
```sh
curl -sSL https://pdm-project.org/install-pdm.py | python3 -
```

### Install development dependencies

```sh
pdm install -dG test,lint,debug
```

### Environment variables

Create a `.env` file according to `.env.example`

### AsyncMartian class and tests

`AsyncMartian` is the main class that uses the LLM services. It receives the
**llm_service and its arguments**. This way, each request using AsyncMartian
can have different arguments according to the LLM services, e.g. for Anthropic:

```python
# See martian/_tests/martian/test_llm_service.py
from martian.client_martian import AsyncMartian
from martian.enums import LLMServiceEnum
from martian.settings import get_api_key

client = AsyncMartian(
	llm_service=LLMServiceEnum.ANTHROPIC,
	api_key=get_api_key(LLMServiceEnum.ANTHROPIC).get_secret_value(),
)
```
We can now call `create()`. The arguments depend on **each llm service**, e.g., for Anthropic:

```python
# It must be in a async function.
response = await client.create(
	model="claude-3-opus-20240229",
	max_tokens=1024,
	messages=[{"role": "user", "content": "Hello, Claude"}],
)
```

You will get the response of the type `Message`; it is the model that **unifies the responses** from OpenAI, Anthropic,
and Together. It is in `martian.models.martian.Message`.

You can still use the original objects from OpenAI, Anthropic, and Together using:
```python
from martian.openai import AsyncOpenAI  # Together uses this.
from martian.anthropic import AsyncAnthropic
```

#### AsyncMartian Tests

You can see end-to-end test examples in `_tests/martian/test_client_martian.py`. There are tests for the 
three models: OpenAI, Anthropic, and Together. You can run the like in the following example:

```bash
# pdm run pytest -ssvv _tests/martian/test_client_martian.py::TestAsyncMartian::<chosen_test>
pdm run pytest -ssvv _tests/martian/test_client_martian.py::TestAsyncMartian::test_create_anthropic
```

Run all the tests with:
```bash
pdm run pytest -ssvv _tests/martian/test_client_martian.py
```

### Run FastAPI app
You can run an FastAPI app

```sh
pdm run uvicorn martian.main:app
```

It has a single endpoint `GET /{llm_service}` and receives query parameters in the format
`martian.models.request.RequestQuery`. This way, there are more restrictions when using the FastAPI app because
the `RequestQuery` model will accept only a subset of the arguments that any LLM service accepts.

`string_messages` is the main difference from using AsyncMartian, it is a JSON string with the LLM services' messages,
e.g. `[{"role": "user", "content": "Hello, world"}]`. I could not find a way to pass a list of dictionaries in the query.

See the tests `_tests/martian/test_main.py` to know how to call the FastAPI app endpoint.

#### Test FastAPI app

`_tests/martian/test_main.py` has tests for the FastAPI app, you don't need to run the FastAPI server
for to run tests, except for the last test called `test_get_llm_message_openai_to_the_http_server` you need
to run the server for it. You can run the tests with:

```bash
# pdm run pytest -ssvv _tests/martian/test_main.py::<chosen_test>
pdm run pytest -ssvv _tests/martian/test_main.py::test_get_llm_message_openai
```

To test many requests to the FastAPI app, change `n_requests_per_llm_service`
of `test_get_many_llm_messages_from_all_llm_services`, and you can use the following example:
```bash
pdm run pytest -ssvv _tests/martian/test_main.py::test_get_many_llm_messages_from_all_llm_services
```

### Add new LLM service models

To add a new LLM service model, you need to do at least:
* Add a new option with the LLM service in the enum `martian.enums.LLMServiceEnum`
* change `martian.client_martian.AsyncMartian`
  * Add a new `case` with a callable the way we already have two.
* Add map function in `martian.maps`.

### Questions and decisions
* Do we need to handle the args for `AsyncMartian` with a common interface/model?
  * At the moment, I kept `**kwargs` to handle the args. This means that we can pass any args to `AsyncMartian`.
    This way, pass the args for each LLM service.
* Do we need to handle the responses with a common interface/model?
  * I will now unify the response from each LLM service in the model `martian.models.martian.Message`.
* Many decisions based on the OpenAI and Anthropic official docs and code's docstrings. There are comments
  in the code to show where the decisions were made.
* Do we need to handle the errors with a common interface/model?
  * It seems to be a good idea to handle the errors with a common interface/model. I am currently using
  fastapi.HTTPException to raise exceptions to the user with status = 400 code,
  there can be improvements to change the codes according to the error.

This way `AsyncMartian` is open for adding args from each LLM service and the responses are unified in `Message`.
Errors are handled with `fastapi.HTTPException`.

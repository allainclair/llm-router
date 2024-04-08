from fastapi import FastAPI

from martian import router_llm

app = FastAPI()

app.include_router(router_llm.router)

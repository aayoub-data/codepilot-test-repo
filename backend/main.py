from fastapi import FastAPI
from datetime import datetime

app = FastAPI(title="Test App")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"message": "Welcome to CodePilot test app"}

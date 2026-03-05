import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import predict, report, chat
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="AI Disease Diagnosis Platform API",
    description="High-accuracy medical image analysis and LLM-powered guidance",
    version="1.0.0"
)

# CORS Configuration
_allowed_origins_env = os.getenv("ALLOWED_ORIGIN", "http://localhost:3000")
_allowed_origins = [o.strip() for o in _allowed_origins_env.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(predict.router, prefix="/api/prediction", tags=["Prediction"])
app.include_router(report.router, prefix="/api/report", tags=["Reports"])
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])


@app.get("/")
async def root():
    return {"message": "AI Disease Diagnosis API is running", "status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

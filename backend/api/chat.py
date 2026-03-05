from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from llm.groq_medical_assistant import get_chat_response

router = APIRouter()


class ChatMessage(BaseModel):
    role: str        # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    diagnosis_context: Optional[str] = None   # e.g. "Brain MRI – Malignant Neoplasm, 72% confidence"


@router.post("/ask")
async def chat_ask(request: ChatRequest):
    """
    Multi-turn follow-up chat powered by Groq LLaMA 3.3 70B.
    Accepts the full conversation history and an optional diagnosis context string
    so the model stays grounded in the current session's scan result.
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided.")

    messages_dicts = [{"role": m.role, "content": m.content} for m in request.messages]

    try:
        reply = await get_chat_response(messages_dicts, request.diagnosis_context)
        return {"role": "assistant", "content": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {type(e).__name__}")

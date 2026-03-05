import os
import asyncio
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

_SYSTEM_PROMPT = """
You are a professional AI medical assistant that helps patients understand their scan results.
Always be empathetic, clear, and responsible. Never claim certainty – always recommend consulting
a certified medical professional. Do not provide harmful or alarming advice.
Use Markdown formatting with clear section headers.
""".strip()


def _call_groq_sync(prediction: str, score: float, age: str) -> str:
    """Synchronous Groq call – run inside asyncio.to_thread to avoid blocking."""
    confidence_pct = f"{score * 100:.1f}%" if isinstance(score, float) else str(score)
    user_prompt = f"""
Scan Result: {prediction}
Confidence: {confidence_pct}
Patient Age: {age if age and age != "Unknown" else "Not provided"}

Please generate a concise, professional report with these sections:
1. **What This Result Means** – Plain-language explanation
2. **Risk Assessment** – Based on the finding and confidence level
3. **Recommended Next Steps** – Immediate actions and specialist referrals
4. **Lifestyle & Prevention Tips** – Practical advice
5. **Diet Suggestions** – Relevant dietary considerations
6. **Important Disclaimer** – Remind the patient this is AI assistance only
""".strip()

    try:
        completion = _client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            max_tokens=900,
            top_p=1,
            stream=False,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return (
            "⚠️ **Medical Guidance Unavailable**\n\n"
            "Our AI assistant is temporarily unavailable. "
            "Please consult a certified healthcare professional for guidance on your scan results.\n\n"
            f"_(Technical note: {type(e).__name__})_"
        )


async def get_medical_guidance(prediction: str, score: float, age: str) -> str:
    """
    Async wrapper around the synchronous Groq call.
    Uses asyncio.to_thread so the FastAPI event loop is never blocked.
    """
    return await asyncio.to_thread(_call_groq_sync, prediction, score, age)


# ─────────────────────────────────────────────────────────────────────────────
# Follow-up Chat
# ─────────────────────────────────────────────────────────────────────────────

_CHAT_SYSTEM_PROMPT = """
You are GoGenix-Med, an empathetic AI medical assistant integrated into a diagnostic platform.
You help patients understand their scan results and answer follow-up questions clearly and safely.
- Keep answers concise but thorough.
- Never claim to replace a doctor. Always recommend professional consultation.
- If the user asks something outside medical scope, politely redirect.
- Use Markdown for formatting (bold, bullet points) when it improves clarity.
""".strip()


def _call_groq_chat_sync(messages: list, diagnosis_context: str | None) -> str:
    """Synchronous multi-turn chat call for Groq."""
    system_content = _CHAT_SYSTEM_PROMPT
    if diagnosis_context:
        system_content += (
            "\n\n---\n**Current Diagnosis Context (from this session):**\n"
            + diagnosis_context
        )

    groq_messages = [{"role": "system", "content": system_content}]
    for m in messages:
        groq_messages.append({"role": m["role"], "content": m["content"]})

    try:
        completion = _client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=groq_messages,
            temperature=0.5,
            max_tokens=700,
            top_p=1,
            stream=False,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return (
            "⚠️ I'm temporarily unavailable. "
            "Please consult your healthcare professional for further guidance.\n\n"
            f"_(Error: {type(e).__name__})_"
        )


async def get_chat_response(messages: list, diagnosis_context: str | None = None) -> str:
    """Async wrapper for multi-turn follow-up chat."""
    return await asyncio.to_thread(_call_groq_chat_sync, messages, diagnosis_context)


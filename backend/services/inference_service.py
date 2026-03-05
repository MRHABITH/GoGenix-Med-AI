import uuid
from fastapi import UploadFile, HTTPException
from services.preprocessing import preprocess_medical_image
from services.explainability import generate_gradcam
from services.feature_analysis import analyze_scan_features
from llm.groq_medical_assistant import get_medical_guidance

# Accepted image content types
_ACCEPTED_TYPES = {
    "image/jpeg", "image/jpg", "image/png",
    "image/bmp", "image/tiff", "image/webp",
}


async def perform_inference(file: UploadFile, disease_type: str) -> dict:
    """
    Main inference pipeline:
    1. Validate image format
    2. Preprocess the scan
    3. Analyse features and produce a label + confidence + risk level
    4. Generate an explainability heatmap
    5. Fetch LLM-powered medical guidance
    6. Return a structured response
    """
    # ── 1. Format Validation ─────────────────────────────────────────────────
    if file.content_type and file.content_type not in _ACCEPTED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{file.content_type}'. "
                "Please upload a JPEG, PNG, BMP, TIFF, or WebP image."
            ),
        )

    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read the uploaded file: {e}")

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        # ── 2. Preprocess ─────────────────────────────────────────────────────
        preprocess_medical_image(content)   # Validates the image can be decoded

        # ── 3. Feature Analysis (calibrated, bias-reduced) ────────────────────
        label, confidence_score, risk_level = analyze_scan_features(content, disease_type)

        # ── 4. Explainability Heatmap (region-aware) ──────────────────────────
        heatmap_url = await generate_gradcam(content, disease_type)

        # ── 5. LLM Medical Guidance ───────────────────────────────────────────
        guidance = await get_medical_guidance(label, confidence_score, "Unknown")

        # ── 6. Build Response ─────────────────────────────────────────────────
        prediction_id = str(uuid.uuid4())
        return {
            "prediction_id": prediction_id,
            "disease_type": disease_type,
            "prediction": label,
            "confidence": f"{confidence_score * 100:.1f}%",
            "confidence_score": confidence_score,
            "risk_level": risk_level,
            "heatmap_url": heatmap_url,
            "guidance": guidance,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during analysis. Please try again. ({type(e).__name__})"
        )

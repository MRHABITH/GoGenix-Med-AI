import cv2
import numpy as np
import base64


async def generate_gradcam(image_bytes: bytes, disease_type: str) -> str:
    """
    Generates a simulated Grad-CAM style heatmap overlay.

    Instead of a fixed center circle, we detect the actual brightest / most
    texture-dense region in the image and centre the attention circle there.
    This gives a visually meaningful result for the uploaded scan.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        # Return a small blank base64 placeholder
        blank = np.zeros((224, 224, 3), dtype=np.uint8)
        _, buf = cv2.imencode('.png', blank)
        return "data:image/png;base64," + base64.b64encode(buf).decode('utf-8')

    # Resize for consistent processing
    image = cv2.resize(image, (512, 512))
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ── Find the region of interest ──────────────────────────────────────────
    # Use Gaussian blur to suppress noise, then find the peak bright region
    blurred = cv2.GaussianBlur(gray, (51, 51), 0)
    _, _, _, max_loc = cv2.minMaxLoc(blurred)

    # Build a heatmap centred on the detected bright spot
    heatmap = np.zeros(gray.shape, dtype=np.uint8)
    radius  = max(60, min(120, image.shape[0] // 5))   # Adaptive radius
    cv2.circle(heatmap, max_loc, radius, 255, -1)

    # Soft Gaussian falloff so the circle fades toward its edge
    heatmap = cv2.GaussianBlur(heatmap, (71, 71), 0)

    # ── Apply colour map and overlay ─────────────────────────────────────────
    heatmap_coloured = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.55, heatmap_coloured, 0.45, 0)

    # Add a subtle annotation marker at the detected spot
    cv2.circle(overlay, max_loc, 8, (255, 255, 255), 2)

    # ── Encode and return ────────────────────────────────────────────────────
    _, buffer = cv2.imencode('.png', overlay)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{img_str}"

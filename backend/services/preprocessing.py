import cv2
import numpy as np
from PIL import Image

def preprocess_medical_image(image_bytes: bytes, target_size=(224, 224)):
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Resize first for consistency
    image = cv2.resize(image, target_size)
    
    # 1. Skull Stripping (Contour-based Brain Masking)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)
        image = cv2.bitwise_and(image, image, mask=mask)
        
    # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization) for better visibility
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # 3. Noise reduction
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    
    # 4. Normalization
    image = image.astype('float32') / 255.0
    
    return image

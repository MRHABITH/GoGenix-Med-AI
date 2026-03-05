from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from services.inference_service import perform_inference

router = APIRouter()

@router.post("/brain")
async def predict_brain(file: UploadFile = File(...)):
    return await perform_inference(file, "brain")

@router.post("/lung")
async def predict_lung(file: UploadFile = File(...)):
    return await perform_inference(file, "lung")

@router.post("/cancer")
async def predict_cancer(file: UploadFile = File(...)):
    return await perform_inference(file, "cancer")

@router.post("/renal")
async def predict_renal(file: UploadFile = File(...)):
    return await perform_inference(file, "renal")

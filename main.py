from typing import Annotated
from typing import Tuple
from fastapi.responses import FileResponse, JSONResponse
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from sources.pdf_read import read_pdf
import numpy as np
import pickle
from sources.model import MyModel
import torch
import torch.nn as nn
import sys
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from typing import List
from transformers import AutoTokenizer
from config.config import Settings
from pydantic import BaseModel

class PredictionResponse(BaseModel):
    filename: str
    class_name: str

settings = Settings()

executor = ThreadPoolExecutor(max_workers=1)

model_lock = threading.Lock()

target_names = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']

tokenizer = AutoTokenizer.from_pretrained(settings.bert_model_name)

model = MyModel(settings.bert_model_name, settings.lstm_units, settings.dense_units, settings.num_classes, settings.nhead, settings.num_layers, settings.dropout)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    weights = torch.load("model/best_model_.pth", map_location=DEVICE)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Failed to load model weights: {str(e)}")

model.load_state_dict(weights)

model.to(DEVICE)

model.eval()

def prepare_file(file: UploadFile) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Read PDF, tokenize it and turn it into input_ids and attention_mask

    Args:
        file (UploadFile): Loaded file

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - input_ids: Tensor of tokens (batch_size, seq_len)
            - attention_mask: Tensors of masks (batch_size, seq_len)
    Raises:
        HTTPException: file is not PDF
    """
    if not file.filename.lower().endswith(".pdf"): raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    text = read_pdf(file.file)
    X_tensor = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    input_ids = X_tensor['input_ids'].to(DEVICE)
    attention_mask = X_tensor['attention_mask'].to(DEVICE)
    return input_ids, attention_mask

async def async_prepare_file(file: UploadFile) -> torch.Tensor:
    """
    Calls the prepare_file asynchronously in a separate thread.

    Args:
        file (UploadFile): Loaded file

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - input_ids: Tensor of tokens (batch_size, seq_len)
            - attention_mask: Tensors of masks (batch_size, seq_len)
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: prepare_file(file))

def classify_pdf(input_ids, attention_mask) -> str:
    """
    Makes a class prediction from an input tensor using a model.

    Args:
        input_ids: Tensor of tokens (batch_size, seq_len)
        attention_mask: Tensors of masks (batch_size, seq_len)

    Returns:
        str: Name of the predicted class

    Raises:
        HTTPException: If an error occurred during inference.
    """
    try:
        preds = model(input_ids=input_ids, attention_mask=attention_mask)
        final_pred = preds.mean(dim=0).detach().cpu().numpy()
        pred_class_index = np.argmax(final_pred)
        pred_class_name = target_names[pred_class_index]
        return pred_class_name
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(error)}")

def model_call(input_ids, attention_mask) -> str:
    """
    Wraps a model call in a thread lock.

    Args:
        input_ids: Tensor of tokens (batch_size, seq_len)
        attention_mask: Tensors of masks (batch_size, seq_len)

    Returns:
        str: Name of the predicted class
    """
    with model_lock:
        return classify_pdf(input_ids, attention_mask)

async def async_classify(file: UploadFile) -> str:
    """
    Asynchronously classify the uploaded PDF file.

    Args:
    file (UploadFile): The uploaded file.

    Returns:
    str: The name of the predicted class
    """
    input_ids, attention_mask = await async_prepare_file(file)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, lambda: model_call(input_ids, attention_mask))

app = FastAPI()

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    Global HTTPException error handler.

    Args:
    request (Request): Request object.
    exc (HTTPException): FastAPI exception.

    Returns:
    JSONResponse: JSON with error description.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "url": str(request.url)
        }
    )

@app.post("/uploadfiles")
async def create_upload_files(files: List[UploadFile]) -> List[PredictionResponse]:
    """
    Endpoint for uploading one or more PDF files and classifying them.

    Args:
        files (List[UploadFile]): List of uploaded PDF files.

    Returns:
        List[PredictionResponse]: List of objects with filename and predicted class.
    """
    tasks = [async_classify(file) for file in files]
    results = await asyncio.gather(*tasks)

    response_list = [
        PredictionResponse(filename=file.filename, class_name=result)
        for file, result in zip(files, results)
    ]
    return response_list


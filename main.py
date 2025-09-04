from typing import Annotated
from fastapi.responses import FileResponse, JSONResponse
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from sources.pdf_read import read_pdf
from sources.text_preprocessing import tokenize_long_text
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

executor = ThreadPoolExecutor(max_workers=1)
model_lock = threading.Lock()

target_names = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']

max_tokens = 10000
output_sequence_length = 200 + 1
lstm_units1 = 256
lstm_units2 = 128
dense_units = 256
dropout = 0.5
dropout1 = 0.5
num_classes = 20

try:
    with open("model/vocab.pkl", "rb") as f:
        vocabulary = pickle.load(f)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Failed to load vocabulary: {str(e)}")

model = MyModel(max_tokens, output_sequence_length, lstm_units1, lstm_units2, dense_units, dropout, dropout1, num_classes)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    weights = torch.load("model/best_model_.pth", map_location=DEVICE)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Failed to load model weights: {str(e)}")

model.load_state_dict(weights)

model.to(DEVICE)

model.eval()

def prepare_file(file: UploadFile) -> torch.Tensor:
    """
    Read PDF, tokenize it and turn it into PyTorch Tensor

    Args:
        file (UploadFile): Loaded file

    Returns:
        torch.Tensor: Tensor with tokenized text

    Raises:
        HTTPException: file is not PDF
    """
    if not file.filename.lower().endswith(".pdf"): raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    text = read_pdf(file.file)
    text_blocks = tokenize_long_text(text, vocabulary, output_sequence_length)
    text_blocks_array = np.array(text_blocks)
    text_blocks_tensor = torch.tensor(text_blocks_array, dtype=torch.long).to(DEVICE)
    return text_blocks_tensor

async def async_prepare_file(file: UploadFile) -> torch.Tensor:
    """
    Calls the prepare_file asynchronously in a separate thread.

    Args:
        file (UploadFile): Loaded file

    Returns:
        torch.Tensor: Tensor with tokenized text
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: prepare_file(file))

def classify_pdf(tensor: torch.Tensor) -> str:
    """
    Makes a class prediction from an input tensor using a model.

    Args:
        tensor (torch.Tensor): Input Tensor

    Returns:
        str: Name of the predicted class

    Raises:
        HTTPException: If an error occurred during inference.
    """
    try:
        preds = model(tensor)
        final_pred = preds.mean(dim=0).detach().cpu().numpy()
        pred_class_index = np.argmax(final_pred)
        pred_class_name = target_names[pred_class_index]
        return pred_class_name
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(error)}")

def model_call(tensor: torch.Tensor) -> str:
    """
    Wraps a model call in a thread lock.

    Args:
        tensor (torch.Tensor): Input Tensor

    Returns:
        str: Name of the predicted class
    """
    with model_lock:
        return classify_pdf(tensor)

async def async_classify(file: UploadFile) -> str:
    """
    Asynchronously classify the uploaded PDF file.

    Args:
    file (UploadFile): The uploaded file.

    Returns:
    str: The name of the predicted class
    """
    tensor = await async_prepare_file(file)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, lambda: model_call(tensor))

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
async def create_upload_files(files: List[UploadFile]) -> JSONResponse:
    """
    Endpoint for uploading one or more PDF files and classifying them.

    Args:
    files (list[UploadFile]): List of uploaded files.

    Returns:
    JSONResponse: Dictionary of {filename: predicted_class}.
    """
    tasks = [async_classify(file) for file in files]
    results = await asyncio.gather(*tasks)
    return JSONResponse({file.filename: result for file, result in zip(files, results)}, status_code=200)


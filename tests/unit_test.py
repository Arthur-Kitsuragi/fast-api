import pytest
from io import BytesIO
import os
import sys
from fastapi import UploadFile, HTTPException
import torch
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_read_pdf(monkeypatch):
    class DummyPage:
        def extract_text(self):
            return "dummy text"

    class DummyPdfReader:
        def __init__(self, file):
            self.pages = [DummyPage()]

    monkeypatch.setattr("sources.pdf_read.PdfReader", DummyPdfReader)

    from sources.pdf_read import read_pdf

    dummy_file = BytesIO(b"not a real pdf")
    text = read_pdf(dummy_file)

    assert text.strip() == "dummy text"

def test_prepare_file_pdf(monkeypatch):
    monkeypatch.setattr("sources.pdf_read.read_pdf", lambda f: "dummy text for tokenizer")

    from main import prepare_file
    dummy_pdf = UploadFile(filename="test.pdf", file=BytesIO(b"not a real pdf"))

    input_ids, attention_mask = prepare_file(dummy_pdf)

    assert isinstance(input_ids, torch.Tensor)
    assert isinstance(attention_mask, torch.Tensor)

def test_prepare_file_not_pdf():
    from main import prepare_file
    dummy_file = UploadFile(filename="test.txt", file=BytesIO(b"hello"))

    with pytest.raises(HTTPException):
        prepare_file(dummy_file)


def test_classify_pdf(monkeypatch):
    class DummyModel:
        def __call__(self, input_ids, attention_mask):
            return torch.tensor([[0.1, 0.9, 0.0]])  # batch_size=1, 3 класса

    monkeypatch.setattr("main.model", DummyModel())
    monkeypatch.setattr("main.target_names", ["class0", "class1", "class2"])

    import main
    input_ids = torch.zeros((1, 5))
    attention_mask = torch.ones((1, 5))

    pred_class = main.classify_pdf(input_ids, attention_mask)

    assert isinstance(pred_class, str)
    assert pred_class == "class1"

def test_classify_pdf_raises(monkeypatch):
    class DummyModel:
        def __call__(self, input_ids, attention_mask):
            raise ValueError("fail")

    monkeypatch.setattr("main.model", DummyModel())
    from main import classify_pdf
    input_ids = torch.zeros((1,5))
    attention_mask = torch.ones((1,5))

    with pytest.raises(HTTPException):
        classify_pdf(input_ids, attention_mask)


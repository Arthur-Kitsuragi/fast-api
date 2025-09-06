from fastapi.testclient import TestClient
import io
from fastapi import UploadFile
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app.main
client = TestClient(app.main.app)

def test_uploadfiles_endpoint(monkeypatch):

    async def dummy_classify(file):
        return "class1"

    monkeypatch.setattr(app.main, "async_classify", dummy_classify)

    dummy_file = io.BytesIO(b"%PDF-1.4 dummy content")

    response = client.post(
        "/uploadfiles",
        files={"files": ("test.pdf", dummy_file, "application/pdf")}
    )

    assert response.status_code == 200
    json_data = response.json()
    assert isinstance(json_data, list)
    assert json_data[0]["filename"] == "test.pdf"
    assert json_data[0]["class_name"] == "class1"
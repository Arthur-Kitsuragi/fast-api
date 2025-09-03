from typing import Annotated
from fastapi.responses import FileResponse
from fastapi import FastAPI, File, UploadFile
from pdf_read import read_pdf
from text_preprocessing import tokenize_long_text
import numpy as np
from tensorflow.keras.models import load_model
import pickle

target_names = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']

best_model_loaded = load_model("best_model_final.keras", compile=False)

output_sequence_length = 201

with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

def func(file):
    try:
        if not file.filename.lower().endswith(".pdf"): return "not pdf"
        text = read_pdf(file.file)
        blocks = tokenize_long_text(text, vocab, output_sequence_length)
        blocks_array = np.array(blocks)
        preds = best_model_loaded.predict(blocks_array, verbose=0)
        final_pred = preds.mean(axis=0)
        pred_class_index = np.argmax(final_pred)
        pred_class_name = target_names[pred_class_index]
        return pred_class_name
    except Exception as e:
        print(f"Error in func for file {file.filename}: {e}")
        return "error"
    
app = FastAPI()

@app.get("/")
def root():
    return FileResponse("public/index.html")

@app.post("/uploadfiles")
def create_upload_files(files: list[UploadFile]):
    d = {}
    for file in files:
        d[file.filename] = func(file)
    return d


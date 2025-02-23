from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
import json
import docx2txt
from pypdf import PdfReader
import io

app = FastAPI()

@app.post("/load_data/")
async def load_data(
    files: List[UploadFile] = File(...)
):
    """
    Endpoint to load data from various file formats (JSON, DOCX, PDF, Text).
    """
    loaded_data = {}
    for file in files:
        file_type = file.content_type
        file_content = None

        try:
            if file_type == "application/json":
                file_content = json.load(io.StringIO(str(await file.read(), 'utf-8'))) # directly load json
                loaded_data[file.filename] = file_content

            elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document": # for docx files
                text = docx2txt.process(file.file)
                file_content = text
                loaded_data[file.filename] = file_content

            elif file_type == "application/pdf":
                pdf_reader = PdfReader(file.file)
                text_content = ""
                for page in pdf_reader.pages:
                    text_content += page.extract_text()
                file_content = text_content
                loaded_data[file.filename] = file_content

            elif file_type == "text/plain": # for text files
                file_content = str(await file.read(), 'utf-8')
                loaded_data[file.filename] = file_content

            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_type} for file {file.filename}")

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing file {file.filename}: {str(e)}")
    
    return {"loaded_files_data": loaded_data}
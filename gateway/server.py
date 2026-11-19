"""
FastAPI wrapper around the verifier logic in ask.py. Exposes POST /ask
so the GUI (or any other tool) can call the router over HTTP.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from docx import Document
from io import BytesIO
from fastapi.responses import Response
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from gateway.ask import ask

app = FastAPI(title="Arbiter")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local tool only; fine for localhost use
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    prompt: str

class AskResponse(BaseModel):
    answer: str
    source: str
    samples_agreed: bool | None

@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest):
    result = ask(req.prompt, verbose=False)
    return AskResponse(**result)

class DownloadRequest(BaseModel):
    prompt: str
    answer: str
    source: str

@app.post("/download")
def download_docx(req: DownloadRequest):
    doc = Document()
    doc.add_heading("Arbiter — Answer", level=1)
    doc.add_heading("Question", level=2)
    doc.add_paragraph(req.prompt)
    doc.add_heading(f"Answer ({req.source})", level=2)
    doc.add_paragraph(req.answer)

    buf = BytesIO()
    doc.save(buf)
    buf.seek(0)

    return Response(
        content=buf.getvalue(),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": "attachment; filename=arbiter-answer.docx"},
    )
    ...

@app.get("/health")
def health():
    return {"status": "ok"}
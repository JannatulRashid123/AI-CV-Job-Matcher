from fastapi import FastAPI, UploadFile, File
from pypdf import PdfReader

app = FastAPI(
    title="AI CV & Job Matcher API"
)


@app.get("/")
def home():
    return {
        "message": "AI CV & Job Matcher API is running!"
    }

@app.post("/upload-cv")
async def upload_cv(file: UploadFile = File(...)):
    contents = await file.read()

    with open("temp_cv.pdf", "wb") as pdf_file:
        pdf_file.write(contents)

    reader = PdfReader("temp_cv.pdf")

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "text": text
    }

from fastapi import FastAPI, UploadFile, File

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
    return {
        "filename": file.filename,
        "content_type": file.content_type
    }

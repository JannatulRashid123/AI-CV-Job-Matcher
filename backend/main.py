from fastapi import FastAPI, UploadFile, File
from pypdf import PdfReader
import re


SKILLS = [
    # Programming
    "Python",
    "Java",
    "JavaScript",
    "TypeScript",
    "C++",
    "C#",
    "SQL",

    # Web development
    "HTML",
    "CSS",
    "React",
    "Node.js",
    "FastAPI",
    "Django",

    # Development tools
    "Git",
    "GitHub",
    "Docker",

    # Cloud
    "AWS",
    "Azure",

    # AI and data
    "Machine Learning",
    "Artificial Intelligence",
    "Data Analysis",
    "TensorFlow",
    "PyTorch",

    # Transferable skills
    "Communication",
    "Problem Solving",
    "Analytical Skills",
    "Analytical Thinking",
    "Organisation",
    "Organisational Skills",
    "Teamwork",
    "Leadership",
    "Customer Service",
    "Student Engagement",
    "Mathematical Skills",
    "Mathematical Instruction"
]

app = FastAPI(
    title="AI CV & Job Matcher API"
)


def extract_skills(text):
    found_skills = []

    for skill in SKILLS:
        pattern = r"\b" + re.escape(skill) + r"\b"

        if re.search(pattern, text, re.IGNORECASE):
            found_skills.append(skill)

    return found_skills

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

    skills = extract_skills(text)

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "text": text,
        "skills": skills
    }
@app.post("/analyse-job")
async def analyse_job(job_description: str):
    job_skills = extract_skills(job_description)

    return {
        "job_description": job_description,
        "skills": job_skills
    }
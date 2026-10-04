from fastapi import FastAPI, UploadFile, File
from pypdf import PdfReader
import re
from sentence_transformers import SentenceTransformer, util


# -----------------------------
# AI MODEL
# -----------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# APP
# -----------------------------

app = FastAPI(title="AI CV & Job Matcher API")


# -----------------------------
# SKILLS LIST
# -----------------------------

SKILLS = [
    "Python",
    "Java",
    "JavaScript",
    "TypeScript",
    "C++",
    "C#",
    "SQL",
    "HTML",
    "CSS",
    "React",
    "Node.js",
    "FastAPI",
    "Django",
    "Git",
    "GitHub",
    "Docker",
    "AWS",
    "Azure",
    "Machine Learning",
    "Artificial Intelligence",
    "Data Analysis",
    "TensorFlow",
    "PyTorch",
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


# -----------------------------
# SKILL EXTRACTION
# -----------------------------

def extract_skills(text):
    found_skills = []

    for skill in SKILLS:
        pattern = r"\b" + re.escape(skill) + r"\b"

        if re.search(pattern, text, re.IGNORECASE):
            found_skills.append(skill)

    return found_skills


# -----------------------------
# HOME PAGE
# -----------------------------

@app.get("/")
def home():
    return {
        "message": "AI CV & Job Matcher API is running!"
    }


# -----------------------------
# CV UPLOAD
# -----------------------------

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


# -----------------------------
# JOB ANALYSIS
# -----------------------------

@app.post("/analyse-job")
async def analyse_job(job_description: str):

    job_skills = extract_skills(job_description)

    return {
        "job_description": job_description,
        "skills": job_skills
    }


# -----------------------------
# CV + JOB MATCHING
# -----------------------------

@app.post("/match")
async def match_cv_to_job(
    cv_text: str,
    job_description: str
):

    # Extract skills
    cv_skills = extract_skills(cv_text)
    job_skills = extract_skills(job_description)

    # Find matched and missing skills
    matched_skills = []
    missing_skills = []

    for skill in job_skills:

        if skill in cv_skills:
            matched_skills.append(skill)
        else:
            missing_skills.append(skill)

    # Calculate keyword match percentage
    if len(job_skills) > 0:

        keyword_match_percentage = round(
            (len(matched_skills) / len(job_skills)) * 100
        )

    else:

        keyword_match_percentage = 0

    # -----------------------------
    # AI SEMANTIC MATCHING
    # -----------------------------

    cv_embedding = model.encode(
        cv_text,
        convert_to_tensor=True
    )

    job_embedding = model.encode(
        job_description,
        convert_to_tensor=True
    )

    similarity = util.cos_sim(
        cv_embedding,
        job_embedding
    )

    ai_match_percentage = round(
        float(similarity[0][0]) * 100,
        2
    )

    # -----------------------------
    # MATCH RATING
    # -----------------------------

    if ai_match_percentage >= 75:

        match_rating = "Strong Match"

    elif ai_match_percentage >= 50:

        match_rating = "Good Match"

    else:

        match_rating = "Weak Match"

    # -----------------------------
    # MATCH EXPLANATION
    # -----------------------------

    if ai_match_percentage >= 75:

        match_summary = (
            "The CV is a strong match for this job. "
            "The candidate's experience and skills are closely "
            "related to the requirements of the role."
        )

    elif ai_match_percentage >= 50:

        match_summary = (
            "The CV is a good match for this job. "
            "The candidate has several relevant skills, but "
            "there are some areas that could be improved."
        )

    else:

        match_summary = (
            "The CV has a weak match with this job. "
            "The candidate should develop more of the skills "
            "required for the role."
        )

    # -----------------------------
    # RECOMMENDATIONS
    # -----------------------------

    recommendations = []

    if missing_skills:

        recommendations.append(
            "Consider developing these skills: "
            + ", ".join(missing_skills)
        )

    if keyword_match_percentage < 50:

        recommendations.append(
            "Add more relevant technical and transferable "
            "skills to your CV."
        )

    if not recommendations:

        recommendations.append(
            "Your CV already covers most of the skills "
            "identified in the job description."
        )

    # -----------------------------
    # RETURN RESULTS
    # -----------------------------

    return {
        "cv_skills": cv_skills,
        "job_skills": job_skills,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "keyword_match_percentage": keyword_match_percentage,
        "ai_match_percentage": ai_match_percentage,
        "match_rating": match_rating,
        "match_summary": match_summary,
        "recommendations": recommendations
    }
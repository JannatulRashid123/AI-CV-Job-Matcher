
import re

import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util


st.set_page_config(
    page_title="AI CV & Job Matcher",
    page_icon="📄",
    layout="wide",
)


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


SKILLS = [
    "Python", "Java", "JavaScript", "TypeScript", "C++", "C#",
    "SQL", "HTML", "CSS", "React", "Node.js", "FastAPI",
    "Django", "Git", "GitHub", "Docker", "AWS", "Azure",
    "Machine Learning", "Artificial Intelligence", "Data Analysis",
    "TensorFlow", "PyTorch", "Communication", "Problem Solving",
    "Analytical Skills", "Analytical Thinking", "Organisation",
    "Organisational Skills", "Teamwork", "Leadership",
    "Customer Service", "Student Engagement", "Mathematical Skills",
    "Mathematical Instruction",
]


def extract_skills(text):
    found_skills = []

    for skill in SKILLS:
        pattern = r"\b" + re.escape(skill) + r"\b"

        if re.search(pattern, text, re.IGNORECASE):
            found_skills.append(skill)

    return found_skills


def extract_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    pages = []

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            pages.append(page_text)

    return "\n".join(pages)


st.title("📄 AI CV & Job Matcher")
st.write(
    "Upload your CV and paste a job description to discover "
    "how closely your skills and experience match the role."
)

st.info(
    "Your CV is processed for this analysis. "
    "This demo does not save uploaded CVs to a database."
)

cv_file = st.file_uploader(
    "Upload your CV as a PDF",
    type=["pdf"],
)

job_description = st.text_area(
    "Paste the job description",
    height=220,
    placeholder="Paste the full job description here...",
)

if st.button("Analyse my CV", type="primary"):
    if cv_file is None:
        st.error("Please upload your CV as a PDF.")
    elif not job_description.strip():
        st.error("Please paste a job description.")
    else:
        try:
            with st.spinner("Reading your CV and loading the AI model..."):
                cv_text = extract_pdf_text(cv_file)

                if not cv_text.strip():
                    st.error(
                        "No readable text was found in the PDF. "
                        "Try a text-based PDF rather than a scanned image."
                    )
                    st.stop()

                cv_skills = extract_skills(cv_text)
                job_skills = extract_skills(job_description)

                matched_skills = [
                    skill for skill in job_skills
                    if skill in cv_skills
                ]
                missing_skills = [
                    skill for skill in job_skills
                    if skill not in cv_skills
                ]

                if job_skills:
                    keyword_percentage = round(
                        len(matched_skills) / len(job_skills) * 100,
                        2,
                    )
                else:
                    keyword_percentage = 0

                model = load_model()

                cv_embedding = model.encode(
                    cv_text,
                    convert_to_tensor=True,
                )
                job_embedding = model.encode(
                    job_description,
                    convert_to_tensor=True,
                )

                similarity = util.cos_sim(
                    cv_embedding,
                    job_embedding,
                )

                ai_percentage = round(
                    float(similarity[0][0].item()) * 100,
                    2,
                )

            st.subheader("Your match results")

            first, second = st.columns(2)

            first.metric(
                "AI semantic match",
                f"{ai_percentage}%",
            )
            second.metric(
                "Keyword skill match",
                f"{keyword_percentage}%",
            )

            if ai_percentage >= 75:
                summary = (
                    "Your CV appears strongly related to this job "
                    "description. Review the required skills to "
                    "confirm that you meet the role's criteria."
                )
            elif ai_percentage >= 50:
                summary = (
                    "Your CV has some relevant similarities to this "
                    "job. Strengthening your evidence of relevant "
                    "skills may improve your application."
                )
            else:
                summary = (
                    "Your CV appears less closely related to this "
                    "job description. Review the requirements and "
                    "identify relevant experience you could highlight."
                )

            st.subheader("Why you matched")
            st.write(summary)

            left, right = st.columns(2)

            with left:
                st.subheader("Matched skills")
                if matched_skills:
                    for skill in matched_skills:
                        st.write(f"✅ {skill}")
                else:
                    st.write("No matching listed skills were found.")

            with right:
                st.subheader("Skills to develop or highlight")
                if missing_skills:
                    for skill in missing_skills:
                        st.write(f"📌 {skill}")
                else:
                    st.write(
                        "No missing listed skills were identified."
                    )

            st.subheader("Skills detected in your CV")
            if cv_skills:
                st.write(", ".join(cv_skills))
            else:
                st.write(
                    "No skills from the current skills list were detected."
                )

            st.subheader("Recommendations")
            if missing_skills:
                st.write(
                    "Consider developing or highlighting these skills: "
                    + ", ".join(missing_skills)
                    + "."
                )

            if keyword_percentage < 50:
                st.write(
                    "Review the job requirements and add relevant "
                    "skills to your CV where you can support them "
                    "with genuine experience or evidence."
                )

            if not missing_skills and keyword_percentage >= 50:
                st.write(
                    "Continue tailoring your CV to the specific "
                    "responsibilities and requirements of the role."
                )

            st.caption(
                "AI similarity is an estimate, not a probability "
                "of getting an interview. Review the results yourself."
            )

        except Exception as error:
            st.error(
                "The analysis could not be completed. "
                "Check that the PDF is valid and try again."
            )
            st.exception(error)

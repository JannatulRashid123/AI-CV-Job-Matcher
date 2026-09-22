from fastapi import FastAPI

app = FastAPI(
    title="AI CV & Job Matcher API"
)


@app.get("/")
def home():
    return {
        "message": "AI CV & Job Matcher API is running!"
    }


from fastapi import FastAPI


app = FastAPI(title="Nossa API com FastAPI")

@app.get("/")
def read_root():
    return{"message": "Hello, fastapi"}
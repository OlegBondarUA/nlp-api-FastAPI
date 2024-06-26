from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Tuple
import nltk
import logging

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

app = FastAPI()

logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class TextRequest(BaseModel):
    text: str


class TokenResponse(BaseModel):
    tokens: List[str]


class POSTagResponse(BaseModel):
    pos_tags: List[Tuple[str, str]]


class NERResponse(BaseModel):
    entities: List[Tuple[str, str]]


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.error(f"Error occurred: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "An internal error occurred. Please try again later."}
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logging.error(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()}
    )


@app.post("/tokenize", response_model=TokenResponse)
def tokenize(request: TextRequest):
    try:
        tokens = nltk.word_tokenize(request.text)
        logging.info("Tokenization successful")
        return {"tokens": tokens}
    except Exception as e:
        logging.error(f"Tokenization error: {e}")
        raise HTTPException(status_code=500, detail="Tokenization failed")


@app.post("/pos_tag", response_model=POSTagResponse)
def pos_tag(request: TextRequest):
    try:
        tokens = nltk.word_tokenize(request.text)
        pos_tags = nltk.pos_tag(tokens)
        logging.info("POS tagging successful")
        return {"pos_tags": pos_tags}
    except Exception as e:
        logging.error(f"POS tagging error: {e}")
        raise HTTPException(status_code=500, detail="POS tagging failed")


@app.post("/ner", response_model=NERResponse)
def ner(request: TextRequest):
    try:
        tokens = nltk.word_tokenize(request.text)
        pos_tags = nltk.pos_tag(tokens)
        chunks = nltk.ne_chunk(pos_tags)

        entities = []
        for chunk in chunks:
            if hasattr(chunk, 'label'):
                entity = " ".join(c[0] for c in chunk)
                entities.append((entity, chunk.label()))

        logging.info("NER successful")
        return {"entities": entities}
    except Exception as e:
        logging.error(f"NER error: {e}")
        raise HTTPException(status_code=500, detail="NER failed")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

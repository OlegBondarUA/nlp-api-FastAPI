from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Tuple
import nltk


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

app = FastAPI()


class TextRequest(BaseModel):
    text: str


class TokenResponse(BaseModel):
    tokens: List[str]


class POSTagResponse(BaseModel):
    pos_tags: List[Tuple[str, str]]


class NERResponse(BaseModel):
    entities: List[Tuple[str, str]]


@app.post("/tokenize", response_model=TokenResponse)
def tokenize(request: TextRequest):
    tokens = nltk.word_tokenize(request.text)
    return {"tokens": tokens}


@app.post("/pos_tag", response_model=POSTagResponse)
def pos_tag(request: TextRequest):
    tokens = nltk.word_tokenize(request.text)
    pos_tags = nltk.pos_tag(tokens)
    return {"pos_tags": pos_tags}


@app.post("/ner", response_model=NERResponse)
def ner(request: TextRequest):
    tokens = nltk.word_tokenize(request.text)
    pos_tags = nltk.pos_tag(tokens)
    chunks = nltk.ne_chunk(pos_tags)

    entities = []
    for chunk in chunks:
        if hasattr(chunk, 'label'):
            entity = " ".join(c[0] for c in chunk)
            entities.append((entity, chunk.label()))

    return {"entities": entities}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

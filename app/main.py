from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import numpy as np

import os
from typing import Optional
import jwt

from app.auth import(create_token, get_current_user, TOKEN_EXPIRE_MINUTES)


SECRET_KEY = os.getenv("JWT_SECRET", "minha-chave-jwt-super-secreta")
ALGORITHM = "HS256"
TOKEN_EXPIRE_MINUTES = 30

USERS_DB={
    "admin": {"password":"admin123", "role":"admin"},
    "admin": {"password":"user123", "role":"user"}

}

class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: TOKEN_EXPIRE_MINUTES * 60

class IrisRequest(BaseModel):
    """
    Modelo de entrada - Define os dados que o cliente deve enviar para rota /predict

    Cada campo tem:
    -tipo(float)
    -Field(...)
    example- valor de exemplo no Swagger

    """

    sepal_length: float = Field(
        ...,
        description="comprimento de sepala (cm)",
        examples=[1.4]
    )

    sepal_width: float = Field(
        ...,
        description="largura da sepala (cm)",
        examples=[3.5]
    )

    petal_length: float = Field(
        ...,
        description="comprimento de petala (cm)",
        examples=[1.6]
    )

    petal_width: float = Field(
        ...,
        description="largura da petala (cm)",
        examples=[0.5]
    )


class IrisPrediction(BaseModel):
    """
    Modelo de predicao - O resultado de classificação

    Contem o nome da classe prevista e seu indice numerico 
    """

    classe: str = Field(..., description="nome da especie prevista")
    classe_indice: int = Field(..., description="Indice numero da classe")


class IrisProbabilities(BaseModel):
    """
    Modelo de probabilidades - Probabilidades por classe
    """

    setosa: float
    versicolor: float
    virginica: float


class IrisResponse(BaseModel):
    """
    Modelo de resposta - O que a API retorna para o cliente
    """

    sucesso: bool
    predicao: IrisPrediction
    probabilidades: IrisProbabilities
    entrada_recebida: IrisRequest


try:
    with open('modelo_iris.pkl', 'rb') as f:
        modelo = pickle.load(f)
    print("OK modelo carregado com sucesso")

    MODELO_CARREGADO = True
    classes = modelo.classes_

except Exception as e:
    print(f"ERRO - Falha ao carregar o modelo: {e}")
    modelo = None
    MODELO_CARREGADO = False
    classes = None

app = FastAPI(
    title="API de clasificação de flores Iris",
    description="API para classificar especies de flores iris utilizando machine learning, evoluindo a nossa api flask",
    version='2.0.0',
    openapi_tags=[
        {"name": "Auth", "description":"Autenticação"},
        {"name": "Predict","description": "Prediçã protegida"}
    ]
)

@app.get("/")

def home():
    """
    Rota principal que retorna informacoes sobre a API.
    Util para verificar se API esta no ar e ver os endpoints disponiveis

    teste curl http://127.0.0.1:8000/
    """
    return {
        "nome": "Api de classificao de flores iris",
        "versao": "2.0.0 FastAPI",
        "descricao": "API para classficicar especies de flores iris",
        "endpoints": {
            "GET /": "informacoes da API",
            "GET /health": "Status de saude da API",
            "GET /docs": "documetnacao interativa no swagger",
            "POST /predict": "fazera previsao"
        },
        "modelo_carregado": MODELO_CARREGADO
    }


app.get("/health")
def health():

    """
    Verifica se a api esta funcionando corretamente
    """

    return {
        "status": "healthy" if MODELO_CARREGADO else "unhealthy",
        "modelo_carregado":MODELO_CARREGADO
    }


@app.post("/predict", response_model=IrisResponse)
def predict(payload: IrisRequest) -> IrisResponse:

    """
    Este é o endepoint que recebe dados de flor e retorna a nossa predição
    """

    if not MODELO_CARREGADO:
        raise HTTPException(
            status_code=503,
            detail="Modelo não foi carregado, verifique seus arquivos .pkl"
        )

    if classes is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo carregado sem classes definidas"
        )

    features = np.array([[
        payload.sepal_length,
        payload.petal_length,
        payload.sepal_width,
        payload.petal_width
    ]])

    predicao_indice = modelo.predict(features)[0]

    probabilidades = modelo.predict_proba(features)[0]

    return IrisResponse(
        sucesso = True,
        predicao = IrisPrediction(
            classe=classes[predicao_indice],
            classe_indice=int(predicao_indice)
        ),
        probabilidades=IrisProbabilities(
            setosa=round(float(probabilidades[0]), 4),
            versicolor=round(float(probabilidades[1]),4),
            virginica=round(float(probabilidades[2]),4)
        ),

        entrada_recebida = payload

    )
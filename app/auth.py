from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import numpy as np

import os
from typing import Optional
import jwt


SECRET_KEY = os.getenv("JWT_SECRET", "minha-chave-jwt-super-secreta")
ALGORITHM = "HS256"
TOKEN_EXPIRE_MINUTES = 30

USERS_DB={
    "admin": {"password":"admin123", "role":"admin"},
    "admin": {"password":"user123", "role":"user"}

}

def create_token(username: str, role:str) -> str:
    """
    Cria um token jwt com expiracao
    """

    expire = datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": username,
        "role": role,
        "exp": expire
    }

    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

security = HTTPBearer()

def get_current_user(token: str= Depends(security)) -> dict:
    """
    Extrai e valida usuario do token jwt
    """

    try:
        payload = jwt.decode(token.credentials, SECRET_KEY, algorithms=ALGORITHM)
        return {"username": payload["sub"], "role":payload["role"]}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="token expirado")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=400, detail="token invalido")
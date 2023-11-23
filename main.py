from fastapi import FastAPI
from routes import router


app = FastAPI()

# Incluir routes
app.include_router(router)
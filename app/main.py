# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from app.routers import stations, parameters, historic, forecast, health
from app.config import settings

def create_app() -> FastAPI:
    app = FastAPI(
        title="MeteoSwiss Weather API",
        version="1.0.0",
        description="Comprehensive REST API for Swiss weather data",

    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    
    # Compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Include routers
    app.include_router(stations.router, prefix="/api", tags=["Stations"])
    app.include_router(parameters.router, prefix="/api", tags=["Parameters"])
  
    app.include_router(forecast.router, prefix="/api", tags=["Forecast"])
    app.include_router(historic.router, prefix="/api", tags=["Historic"])
    app.include_router(health.router)

    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "version": "1.0.0",
            "name": "MeteoSwiss Weather API",
        }
    
    return app

app = create_app()

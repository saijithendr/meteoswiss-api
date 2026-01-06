# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZIPMiddleware
from app.routers import stations, parameters, weather, health
from app.config import settings

def create_app() -> FastAPI:
    app = FastAPI(
        title="MeteoSwiss Weather API",
        version="1.0.0",
        description="Comprehensive REST API for Swiss weather data",
        openapi_url="/api/v1/openapi.json",
        docs_url="/api/v1/docs",
        redoc_url="/api/v1/redoc"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    
    # Compression
    app.add_middleware(GZIPMiddleware, minimum_size=1000)
    
    # Include routers
    app.include_router(stations.router, prefix="/api/v1", tags=["Stations"])
    app.include_router(parameters.router, prefix="/api/v1", tags=["Parameters"])
    app.include_router(weather.router, prefix="/api/v1", tags=["Weather"])
    app.include_router(health.router, prefix="/api/v1", tags=["Health"])
    
    # Root endpoint
    @app.get("/api/v1/")
    async def root():
        return {
            "version": "1.0.0",
            "name": "MeteoSwiss Weather API",
            "docs": "/api/v1/docs",
            "redoc": "/api/v1/redoc"
        }
    
    return app

app = create_app()

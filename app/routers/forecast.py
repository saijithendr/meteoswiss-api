# app/routers/forecast.py
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional

from app.services.weather_service import LocalForecastHandler
from app.dependents import get_forecast_handler

router = APIRouter(prefix="/forecast", tags=["forecast"])

@router.get("/parameters")
def forecast_parameters(handler: LocalForecastHandler = Depends(get_forecast_handler)) -> List[str]:
    return handler.get_all_parameters()

@router.get("/point/{pointid}")
def forecast_by_pointid(
    pointid: int,
    parameters: Optional[List[str]] = Query(None),
    latestonly: bool = True,
    filteron: str = Query("measured"),
    forcereload: bool = False,
    handler: LocalForecastHandler = Depends(get_forecast_handler),
):
    result = handler.get_forecast_for_point_id(
        point_id=pointid,
        parameters=parameters,
        latest_only=latestonly,
        filter_on=filteron,
        force_reload=forcereload,
    )
    if not result:
        raise HTTPException(status_code=404, detail="No forecast data found")
    return {
        **result.__dict__,
        "forecastreferencetime": result.forecastreferencetime.isoformat() if result.forecastreferencetime else None,
        "data": result.data.to_dict(orient="records"),
    }

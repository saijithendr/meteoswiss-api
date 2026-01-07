# app/routers/forecast.py
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
import numpy as np

from app.services.weather_service import LocalForecastHandler
from app.dependents import get_forecast_handler

router = APIRouter(prefix="/forecast", tags=["forecast"])

@router.get("/parameters")
def forecast_parameters(handler: LocalForecastHandler = Depends(get_forecast_handler)) -> List[str]:
    return handler.get_all_parameters()

@router.get("/station/{stationname}")
def forecast_by_stationname(
    stationname: str,
    parameters: List[str] = Query(default=[ "dkl010h0",
                                                        "fu3010h0",
                                                        "fu3010h1",
                                                        "fu3q10h0"]),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=1000, description="Items per page"),
    handler: LocalForecastHandler = Depends(get_forecast_handler),
):
    result = handler.get_forecast_for_station_name(
        station_name=stationname,
        parameters=parameters,
    )
    if not result:
        raise HTTPException(status_code=404, detail="No forecast data found")
    clean_df = result.data.replace([np.nan, np.inf, -np.inf], [None, None, None])
    
    # Pagination Logic
    total_items = len(clean_df)
    total_pages = (total_items + page_size - 1) // page_size
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    
    paginated_data = clean_df.iloc[start_idx:end_idx].to_dict(orient="records")

    return {
        **result.__dict__,
        "forecastreferencetime": result.forecast_reference_time.isoformat() if result.forecast_reference_time else None,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_items": total_items,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1
        },
        "data": paginated_data,
    }

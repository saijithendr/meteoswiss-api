# app/routers/historic.py
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
import numpy as np
from app.services.weather_service import HistoricWeatherHandler
from app.dependents import get_historic_handler

router = APIRouter(prefix="/historic", tags=["historic"])

@router.get("/parameters")
def historic_parameters(handler: HistoricWeatherHandler = Depends(get_historic_handler)):
    return handler.list_available_parameters()

@router.get("/station/{stationid}")
def historic_by_stationid(
    stationid: str,
    startdate: Optional[str] = Query(..., description="YYYY-MM-DD HH:MM:SS"),
    enddate: Optional[str] = Query(..., description="YYYY-MM-DD HH:MM:SS"),
    aggregation: str = Query("daily"),
    parameters: Optional[List[str]] = Query(None),
    renamecolumns: bool = True,
    includeunits: bool = True,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=1000, description="Items per page"),
    handler: HistoricWeatherHandler = Depends(get_historic_handler),
):
    result = handler.get_historic_by_station_id(
        station_id=stationid,
        start_date=startdate,
        end_date=enddate,
        aggregation=aggregation,
        parameters=parameters,
        rename_columns=renamecolumns,
        include_units=includeunits,
    )
    if not result:
        raise HTTPException(status_code=404, detail="No historic data found")
    
    clean_df = result.data.replace([np.nan, np.inf, -np.inf], [None, None, None])
    
    # Pagination Logic
    total_items = len(clean_df)
    total_pages = (total_items + page_size - 1) // page_size
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    
    paginated_data = clean_df.iloc[start_idx:end_idx].to_dict(orient="records")
    
    return {
        **result.__dict__,
        "startdate": result.start_date.isoformat() if result.start_date is not None else None,
        "enddate": result.end_date.isoformat() if result.end_date is not None else None,
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

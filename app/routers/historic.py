# app/routers/historic.py
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional

from app.services.weather_service import HistoricWeatherHandler
from app.dependents import get_historic_handler

router = APIRouter(prefix="/historic", tags=["historic"])

@router.get("/parameters")
def historic_parameters(handler: HistoricWeatherHandler = Depends(get_historic_handler)):
    return handler.listavailableparameters()

@router.get("/station/{stationid}")
def historic_by_stationid(
    stationid: str,
    startdate: Optional[str] = Query(None),
    enddate: Optional[str] = Query(None),
    aggregation: str = Query("daily"),
    parameters: Optional[List[str]] = Query(None),
    renamecolumns: bool = True,
    includeunits: bool = True,
    handler: HistoricWeatherHandler = Depends(get_historic_handler),
):
    result = handler.gethistoricbystationid(
        stationid=stationid,
        startdate=startdate,
        enddate=enddate,
        aggregation=aggregation,
        parameters=parameters,
        renamecolumns=renamecolumns,
        includeunits=includeunits,
    )
    if not result:
        raise HTTPException(status_code=404, detail="No historic data found")
    return {
        **result.__dict__,
        "startdate": result.startdate.isoformat() if result.startdate is not None else None,
        "enddate": result.enddate.isoformat() if result.enddate is not None else None,
        "data": result.data.to_dict(orient="records"),
    }

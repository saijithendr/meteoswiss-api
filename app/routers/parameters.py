# app/routers/parameters.py
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Any, Dict, List, Optional

from app.services.parameters_service import MetaParametersLoader
from app.dependents import get_params

router = APIRouter(prefix="/parameters", tags=["parameters"])

@router.get("/sources")
def list_sources(params: MetaParametersLoader = Depends(get_params)) -> List[str]:
    return params.list_sources()

@router.get("")
def list_parameters(
    source: Optional[str] = None,
    forcereload: bool = False,
    params: MetaParametersLoader = Depends(get_params),
):
    df = params.get_all_params(source=source, force_reload=forcereload)
    return df.to_dict(orient="records")

@router.get("/{key}")
def get_parameter(key: str, source: Optional[str] = None, params: MetaParametersLoader = Depends(get_params)):
    meta = params.get(key, source=source)
    if not meta:
        raise HTTPException(status_code=404, detail="Parameter not found")
    return {"source": meta.source, "data": dict(meta.data)}

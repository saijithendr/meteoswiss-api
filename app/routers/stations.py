from fastapi import APIRouter, Query, HTTPException
#from app.models.stations import StationResponse, StationListResponse
from app.services.stations_service import SwissWeatherStations

router = APIRouter(prefix="/stations")
service = SwissWeatherStations()

@router.get("/")
async def list_stations(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    sort_by: str = Query("name"),
    order: str = Query("asc")
):
    """List all Swiss weather stations with pagination."""
    return await service.get_all_stations(limit, offset, sort_by, order)

@router.get("/{station_id}")
async def get_station(station_id: str):
    """Get detailed information for a specific weather station."""
    station = await service.get_by_id(point_id=station_id)
    if not station:
        raise HTTPException(status_code=404, detail="Station not found")
    return station

@router.get("/search")
async def search_stations(
    q: str = Query(...),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """Search stations by name or properties."""
    return await service.search(q, limit, offset)

@router.get("/nearby")
async def find_nearby(
    latitude: float = Query(...),
    longitude: float = Query(...),
    radius_km: float = Query(20, ge=1),
    limit: int = Query(10, ge=1, le=100)
):
    """Find stations within a geographic radius."""
    return await service.find_nearby(latitude, longitude, radius_km, limit)

@router.get("/nearest")
async def find_nearest(
    latitude: float = Query(...),
    longitude: float = Query(...),
    n: int = Query(5, ge=1, le=50)
):
    """Find N nearest stations to a given location."""
    return await service.find_nearest(latitude, longitude, n)

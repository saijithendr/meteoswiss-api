# app/deps.py
from app.services.stations_service import SwissWeatherStations
from app.services.parameters_service import MetaParametersLoader
from app.services.weather_service import LocalForecastHandler, MeteoSwissClient, HistoricWeatherHandler

stations = SwissWeatherStations()
params = MetaParametersLoader()

forecast_handler = LocalForecastHandler(
    stations_handler=stations,
    params_loader=params,
    auto_load_metadata=True,
)

client = MeteoSwissClient()

historic_handler = HistoricWeatherHandler(
    stations_handler=stations,
    params_loader=params,
    meteoswiss_client=client,
)

def get_stations() -> SwissWeatherStations:
    return stations

def get_params() -> MetaParametersLoader:
    return params

def get_forecast_handler() -> LocalForecastHandler:
    return forecast_handler

def get_historic_handler() -> HistoricWeatherHandler:
    return historic_handler

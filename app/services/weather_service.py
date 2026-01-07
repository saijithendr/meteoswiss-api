from __future__ import annotations

from typing import Any, ClassVar

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from io import StringIO

import pandas as pd
import requests

logger = logging.getLogger(__name__)


@dataclass
class ForecastQueryResult:
    """Container for forecast data + metadata."""

    point_id: int
    station_name: str
    latitude: float
    longitude: float
    elevation: float
    station_type: str
    forecast_reference_time: datetime | None
    parameters: list[str]
    data: pd.DataFrame

    def summary(self) -> str:
        """Get result summary."""
        lines = [
            f"Forecast for {self.station_name} (Point ID: {self.point_id})",
            f"  Location: {self.latitude:.6f}°N, {self.longitude:.6f}°E",
            f"  Elevation: {self.elevation:.0f}m",
            f"  Type: {self.station_type}",
            f"  Reference Time: {self.forecast_reference_time}",
            f"  Parameters: {len(self.parameters)}",
            f"  Data Points: {len(self.data)}",
        ]
        if "Date" in self.data.columns and len(self.data) > 0:
            lines.append(
                f"  Date Range: {self.data['Date'].min()} to {self.data['Date'].max()}"
            )
        return "\n".join(lines)


class LocalForecastHandler:
    """
    Handler for MeteoSwiss local forecast data.

    Provides high-level interface to fetch all weather data for specific point_ids.

    """

    STAC_BASE_URL = "https://data.geo.admin.ch/api/stac/v1"
    COLLECTION_ID = "ch.meteoschweiz.ogd-local-forecasting"

    def __init__(
        self,
        stations_handler=None,
        params_loader=None,
        language: str = "en",
        timeout: int = 60,
        auto_load_metadata: bool = True,
    ):
        """
        Initialize the forecast handler.

        Args:
            stations_handler: SwissWeatherStations instance (or None to create new)
            params_loader: MetaParametersLoader instance (or None to create new)
            language: Language for parameter descriptions ('en', 'de', 'fr', 'it')
            timeout: HTTP timeout in seconds
            auto_load_metadata: If True, load parameter and station metadata on init
        """
        self.language = language
        self.timeout = timeout

        # Initialize core components
        if stations_handler is None:
            from stations_service import SwissWeatherStations

            self.stations = SwissWeatherStations()
        else:
            self.stations = stations_handler

        if params_loader is None:
            from parameters_service import MetaParametersLoader

            self.params_loader = MetaParametersLoader()
        else:
            self.params_loader = params_loader

        # Cache
        self._forecast_items_cache = None
        self._parameters_cache = None

        if auto_load_metadata:
            self._init_metadata()

    def _init_metadata(self) -> None:
        """Load parameter and station metadata."""
        try:
            # Load parameter metadata
            logger.info("Loading parameter metadata...")
            self.params_loader.add_source(
                name="forecast_parameters",
                url="https://data.geo.admin.ch/ch.meteoschweiz.ogd-local-forecasting/ogd-local-forecasting_meta_parameters.csv",
                description="MeteoSwiss local forecast parameters",
                encoding="latin-1",
                delimiter=";",
                key_column="parameter_shortname",
            )    

            self.params_loader.add_source(
                name="historic_parameters",
                url="https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/ogd-smn_meta_parameters.csv",
                description="MeteoSwiss Historic parameters",
                encoding="latin-1",
                delimiter=";",
                key_column="point_id",
            )
            self.params_loader.load_source("forecast_parameters")
            self.params_loader.load_source("historic_parameters")
           
            logger.info("✓ Parameter metadata loaded")
        except Exception as e:
            logger.warning(f"Could not load parameter metadata: {e}")

        try:
            logger.info("Loading Swiss weather stations...")
            self.stations.load()
            logger.info("✓ Loaded stations")
        except Exception as e:
            logger.warning(f"Could not load stations: {e}")

    def _get_stac_items(
        self,
        force_reload: bool = False,
        limit: int = 10,
    ) -> list[dict]:
        """
        Get forecast items from STAC API.

        Args:
            force_reload: Force reload even if cached
            limit: Maximum number of items to return

        Returns:
            list of STAC features (forecast items)
        """
        if self._forecast_items_cache and not force_reload:
            return self._forecast_items_cache

        try:
            stac_url = f"{self.STAC_BASE_URL}/collections/{self.COLLECTION_ID}/items"
            logger.info(f"Querying STAC API: {stac_url}")

            response = requests.get(
                stac_url, params={"limit": limit}, timeout=self.timeout
            )
            response.raise_for_status()

            data = response.json()
            items = data.get("features", [])

            logger.info(f"✓ Found {len(items)} forecast items")
            self._forecast_items_cache = items

            return items
        except Exception as e:
            logger.error(f"Failed to query STAC API: {e}")
            return []

    def _find_asset_for_parameter(
        self,
        items: list[dict],
        parameter: str,
    ) -> tuple[str, dict] | None:
        """
        Find CSV asset URL for a parameter.

        Args:
            items: STAC items
            parameter: Parameter code (e.g., 'tre200h0')

        Returns:
            tuple of (asset_name, asset_info) or None
        """
        for item in items:
            assets = item.get("assets", {})

            for asset_name, asset_info in assets.items():
                if asset_info.get("type") == "text/csv" and parameter in asset_name:
                    return (asset_name, asset_info)

        return None

    def _download_parameter_csv(
        self,
        asset_url: str,
        parameter: str,
    ) -> pd.DataFrame | None:
        """
        Download and parse parameter CSV from STAC asset.

        Args:
            asset_url: URL to CSV file
            parameter: Parameter code (for logging)

        Returns:
            DataFrame or None if download fails
        """
        try:
            logger.info(f"Downloading {parameter} from {asset_url[:60]}...")

            response = requests.get(asset_url, timeout=self.timeout)
            response.raise_for_status()

            content = response.content.decode("latin-1")
            df = pd.read_csv(StringIO(content), sep=";")

            logger.info(
                f"✓ Downloaded {parameter}: {len(df)} rows, {len(df.columns)} columns"
            )

            return df
        except Exception as e:
            logger.error(f"Failed to download {parameter}: {e}")
            return None

    def _extract_forecast_reference_time(
        self,
        items: list[dict],
    ) -> datetime | None:
        """
        Extract forecast reference time from STAC item.

        Args:
            items: STAC items

        Returns:
            datetime or None
        """
        try:
            if not items:
                return None

            # Get asset name from first item
            first_item = items[0]
            assets = first_item.get("assets", {})

            if not assets:
                return None

            asset_name = next(iter(assets.keys()))

            # Extract timestamp: vnut12.lssw.202512162300.parameter.csv
            # Pattern: vnut12.lssw.YYYYMMDDHHMM.parameter.csv
            match = re.search(r"\.(\d{12})\.", asset_name)
            if match:
                timestamp_str = match.group(1)
                ref_time = pd.to_datetime(timestamp_str, format="%Y%m%d%H%M")
                return ref_time
        except Exception as e:
            logger.warning(f"Could not extract forecast reference time: {e}")

        return None

    def _get_station_metadata(self, point_id: int) -> dict[str, Any]:
        """
        Get station metadata for a point_id.

        Args:
            point_id: Location ID

        Returns:
            dictionary with station metadata
        """
        try:
            station = self.stations.get_by_id(point_id)
            if station:
                return {
                    "station_name": station.point_name,
                    "latitude": station.coordinates_wgs84_lat,
                    "longitude": station.coordinates_wgs84_lon,
                    "elevation": station.point_height_masl,
                    "station_type": station.point_type,
                }
        except Exception as e:
            logger.warning(f"Could not get metadata for point_id {point_id}: {e}")

        # Fallback
        return {
            "station_name": f"Point {point_id}",
            "latitude": float("nan"),
            "longitude": float("nan"),
            "elevation": float("nan"),
            "station_type": "Unknown",
        }

    def get_all_parameters(self) -> list[str]:
        """
        Get list of all available forecast parameters.

        Returns:
            list of parameter codes
        """
        try:
            if self._parameters_cache:
                return self._parameters_cache

            df_params = self.params_loader.get_all_params(source="forecast_parameters")

            if df_params is not None and not df_params.empty:
                params = (
                    df_params["parameter_shortname"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )
                self._parameters_cache = params
                return params
        except Exception as e:
            logger.warning(f"Could not get parameters list: {e}")

        return []

    def get_forecast_for_point_id(
        self,
        point_id: int,
        parameters: list[str] | None = None,
        latest_only: bool = True,
        filter_on: str = "measured",
        force_reload: bool = False,
    ) -> ForecastQueryResult | None:
        """
        Get all forecast data for a specific point_id.

        Args:
            point_id: Location ID (1-965800)
            parameters: list of parameter codes to fetch. If None, fetch all available.
            force_reload: Force reload even if cached

        Returns:
            ForecastQueryResult with data and metadata

        Example:
            handler = LocalForecastHandler()

            result = handler.get_forecast_for_point_id(
                point_id=1,
                parameters=['tre200h0', 'rre150h0', 'fu3010h0']
            )

            if result:
                print(result.summary())
                print(result.data.head())
        """
        logger.info(f"Fetching forecast for point_id {point_id}")

        # Get STAC items
        items = self._get_stac_items(force_reload=force_reload)
        if not items:
            logger.error("No STAC items found")
            return None

        if latest_only:
            total_items_date = [
                pd.to_datetime(
                    int(prop.get("id").split("-")[0]), format="%Y%m%d", errors="coerce"
                )
                for prop in items
            ]
            today = pd.Timestamp.today().normalize()

            today_index = [
                i for i, date in enumerate(total_items_date) if date == today
            ]

            items = [items[i] for i in today_index]

        # Determine which parameters to fetch
        if parameters is None:
            parameters = self.get_all_parameters()
            logger.info(f"Fetching all {len(parameters)} available parameters")
        else:
            logger.info(f"Fetching {len(parameters)} specified parameters")

        # Download data for each parameter
        all_dfs = []
        for param in parameters:
            # Find asset
            asset_info = self._find_asset_for_parameter(items, param)
            if not asset_info:
                logger.warning(f"Asset not found for parameter {param}")
                continue

            _, asset_data = asset_info
            asset_url = asset_data.get("href")

            # Download CSV
            df = self._download_parameter_csv(asset_url, param)
            if df is None:
                continue

            # Filter for point_id
            df_filtered = df[df["point_id"] == point_id]

            if df_filtered.empty:
                logger.warning(f"No data for point_id {point_id} in parameter {param}")
                continue

            all_dfs.append(df_filtered)

        if not all_dfs:
            logger.error(f"No data found for point_id {point_id}")
            return None

        # Combine data
        combined_df = pd.concat(all_dfs, ignore_index=True)

        if filter_on == "measured":
            combined_df = combined_df[combined_df["point_type_id"] == 3]
        elif filter_on == "observed":
            combined_df = combined_df[combined_df["point_type_id"] == 1]
        else:
            combined_df = combined_df

        combined_df = combined_df.groupby(
            ["point_id", "point_type_id", "Date"], as_index=False
        ).max()

        if "Date" in combined_df.columns:
            try:
                combined_df["Date"] = pd.to_datetime(
                    combined_df["Date"], format="%Y%m%d%H%M", errors="coerce"
                )
                combined_df = combined_df.sort_values("Date").reset_index(drop=True)
            except Exception as e:
                logger.warning(f"Could not parse dates: {e}")

        # Get station metadata
        station_meta = self._get_station_metadata(point_id)

        # Extract forecast reference time
        ref_time = self._extract_forecast_reference_time(items)

        # Create result
        result = ForecastQueryResult(
            point_id=point_id,
            station_name=station_meta["station_name"],
            latitude=station_meta["latitude"],
            longitude=station_meta["longitude"],
            elevation=station_meta["elevation"],
            station_type=station_meta["station_type"],
            forecast_reference_time=ref_time,
            parameters=parameters,
            data=combined_df,
        )

        logger.info(f"✓ Retrieved forecast for {result.station_name}")

        return result

    def get_forecast_for_station_name(
        self,
        station_name: str,
        parameters: list[str] | None = None,
        exact: bool = True,
    ) -> ForecastQueryResult | None:
        """
        Get forecast for a station by name.

        Args:
            station_name: Swiss station name (e.g., 'Grenchen')
            parameters: list of parameter codes
            exact: If True, require exact name match

        Returns:
            ForecastQueryResult or None
        """
        try:
            # Find station
            candidates = self.stations.get_by_name(station_name.lower(), exact=exact)

            if not candidates:
                logger.error(f"Station '{station_name}' not found")
                return None

            if len(candidates) > 1:
                logger.warning(
                    f"Found {len(candidates)} stations with name '{station_name}'"
                )
                logger.info(f"Using first: {candidates[0].point_name}")

            station = candidates[0]

            # Fetch forecast
            return self.get_forecast_for_point_id(
                point_id=station.point_id,
                parameters=parameters,
            )
        except Exception as e:
            logger.error(f"Failed to get forecast for station '{station_name}': {e}")
            return None

    def get_forecast_for_coordinates(
        self,
        lat: float,
        lon: float,
        parameters: list[str] | None = None,
    ) -> ForecastQueryResult | None:
        """
        Get forecast for the nearest station to given coordinates.

        Args:
            lat: Latitude
            lon: Longitude
            parameters: list of parameter codes

        Returns:
            ForecastQueryResult for nearest station
        """
        try:
            # Find nearest station
            nearest = self.stations.find_nearest(lat, lon, n=1)
            if not nearest:
                logger.error("No stations found")
                return None

            station, distance = nearest[0]
            logger.info(
                f"Nearest station: {station.point_name} ({distance:.1f}km away)"
            )

            # Fetch forecast
            return self.get_forecast_for_point_id(
                point_id=station.point_id,
                parameters=parameters,
            )
        except Exception as e:
            logger.error(f"Failed to get forecast for coordinates: {e}")
            return None

    def export_to_csv(
        self,
        result: ForecastQueryResult,
        filepath: str,
        encoding: str = "utf-8",
    ) -> bool:
        """Export forecast data to CSV."""
        try:
            result.data.to_csv(filepath, index=False, encoding=encoding, sep=",")
            logger.info(f"✓ Exported to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to export to CSV: {e}")
            return False

    def export_to_json(
        self,
        result: ForecastQueryResult,
        filepath: str,
    ) -> bool:
        """Export forecast data to JSON."""
        try:
            result.data.to_json(filepath, orient="records", indent=2)
            logger.info(f"✓ Exported to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to export to JSON: {e}")
            return False


@dataclass
class Station:
    """Represents a MeteoSwiss weather station."""

    station_id: str
    name: str
    canton: str
    latitude: float
    longitude: float
    altitude: int


@dataclass
class HistoricQueryResult:
    """Container for historic data + metadata."""

    station_id: str
    station_name: str
    canton: str | None
    latitude: float
    longitude: float
    altitude: float
    aggregation: str
    parameters: list[str]
    start_date: pd.Timestamp | None
    end_date: pd.Timestamp | None
    data: pd.DataFrame


class MeteoSwissClient:
    """
    Client for MeteoSwiss Open Data via STAC API

    Features:
    - Automatic column renaming with meaningful names
    - Downloads parameter metadata from MeteoSwiss
    - Supports multiple languages for column names
    """

    STAC_API_URL = "https://data.geo.admin.ch/api/stac/v1"
    COLLECTION_ID = "ch.meteoschweiz.ogd-smn"
    PARAMETER_METADATA_URL = (
        "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/ogd-smn_meta_parameters.csv"
    )

    HARDCODED_STATIONS: ClassVar[dict[str, Station]] = {
        "GRE": Station(
            station_id="GRE",
            name="Grenchen",
            canton="Solothurn",
            latitude=47.20,
            longitude=7.50,
            altitude=430,
        ),
        "SBE": Station(
            station_id="SBE",
            name="Sebastiansbad",
            canton="Solothurn",
            latitude=47.26,
            longitude=7.64,
            altitude=514,
        ),
    }

    COMMON_PARAMETERS: ClassVar[dict[str, dict[str, str]]] = {
        "tre200s0": {"name": "Air temperature 2m", "unit": "°C", "resolution": "10m"},
        "tre200d0": {
            "name": "Daily mean temperature 2m",
            "unit": "°C",
            "resolution": "daily",
        },
        "tre200dn": {
            "name": "Daily min temperature 2m",
            "unit": "°C",
            "resolution": "daily",
        },
        "tre200dx": {
            "name": "Daily max temperature 2m",
            "unit": "°C",
            "resolution": "daily",
        },
        "rre150z0": {
            "name": "Precipitation 10-min total",
            "unit": "mm",
            "resolution": "10m",
        },
        "rre150d0": {
            "name": "Daily precipitation",
            "unit": "mm",
            "resolution": "daily",
        },
        "fu3010z0": {
            "name": "Wind speed scalar 10m",
            "unit": "km/h",
            "resolution": "10m",
        },
        "dv1010z0": {"name": "Wind direction 10m", "unit": "°", "resolution": "10m"},
        "gust010z0": {"name": "Wind gust peak", "unit": "km/h", "resolution": "10m"},
        "sre000z0": {
            "name": "Sunshine duration 10-min",
            "unit": "min",
            "resolution": "10m",
        },
        "sre000d0": {
            "name": "Daily sunshine duration",
            "unit": "h",
            "resolution": "daily",
        },
        "gre000z0": {
            "name": "Global radiation 10-min mean",
            "unit": "W/m²",
            "resolution": "10m",
        },
        "gre000d0": {
            "name": "Daily global radiation",
            "unit": "MJ/m²",
            "resolution": "daily",
        },
        "prestos0": {
            "name": "Atmospheric pressure",
            "unit": "hPa",
            "resolution": "10m",
        },
        "uty200s0": {"name": "Relative humidity 2m", "unit": "%", "resolution": "10m"},
    }

    PARAMETER_DEFINITIONS: ClassVar[dict[str, dict[str, str]]] = {
        # Temperature parameters
        "tre200": {
            "name_en": "Air temperature 2m",
            "name_de": "Lufttemperatur 2m",
            "unit": "°C",
        },
        "tre005": {
            "name_en": "Air temperature 5cm",
            "name_de": "Lufttemperatur 5cm",
            "unit": "°C",
        },
        "tde200": {"name_en": "Dew point 2m", "name_de": "Taupunkt 2m", "unit": "°C"},
        # Precipitation parameters
        "rre150": {"name_en": "Precipitation", "name_de": "Niederschlag", "unit": "mm"},
        # Wind parameters
        "fu3010": {
            "name_en": "Wind speed scalar",
            "name_de": "Windgeschwindigkeit skalar",
            "unit": "m/s",
        },
        "fkl010": {
            "name_en": "Wind speed",
            "name_de": "Windgeschwindigkeit",
            "unit": "m/s",
        },
        "dkl010": {"name_en": "Wind direction", "name_de": "Windrichtung", "unit": "°"},
        "fve010": {
            "name_en": "Wind speed vector",
            "name_de": "Windgeschwindigkeit vektoriell",
            "unit": "m/s",
        },
        # Gust parameters
        "gust010": {
            "name_en": "Wind gust peak",
            "name_de": "Windböenspitze",
            "unit": "m/s",
        },
        # Sunshine parameters
        "sre000": {
            "name_en": "Sunshine duration",
            "name_de": "Sonnenscheindauer",
            "unit": "min",
        },
        # Radiation parameters
        "gre000": {
            "name_en": "Global radiation",
            "name_de": "Globalstrahlung",
            "unit": "W/m²",
        },
        "ure200": {
            "name_en": "Longwave radiation",
            "name_de": "Langwellige Strahlung",
            "unit": "W/m²",
        },
        "oli000": {
            "name_en": "Longwave incoming radiation",
            "name_de": "Langwellige eingehende Strahlung",
            "unit": "W/m²",
        },
        "olo000": {
            "name_en": "Longwave outgoing radiation",
            "name_de": "Langwellige ausgehende Strahlung",
            "unit": "W/m²",
        },
        "osr000": {
            "name_en": "Shortwave reflected radiation",
            "name_de": "Kurzwellige reflektierte Strahlung",
            "unit": "W/m²",
        },
        "ods000": {
            "name_en": "Diffuse radiation",
            "name_de": "Diffuse Strahlung",
            "unit": "W/m²",
        },
        "erefao": {
            "name_en": "Reference evapotranspiration",
            "name_de": "Referenzverdunstung",
            "unit": "mm",
        },
        # Pressure parameters
        "presta": {
            "name_en": "Atmospheric pressure QFE",
            "name_de": "Luftdruck QFE",
            "unit": "hPa",
        },
        "pp0qff": {
            "name_en": "Pressure QFF (sea level)",
            "name_de": "Luftdruck QFF (Meeresniveau)",
            "unit": "hPa",
        },
        "pp0qnh": {
            "name_en": "Pressure QNH (sea level)",
            "name_de": "Luftdruck QNH (Meeresniveau)",
            "unit": "hPa",
        },
        "ppz850": {
            "name_en": "Pressure at 850 hPa",
            "name_de": "Druck auf 850 hPa",
            "unit": "hPa",
        },
        "ppz700": {
            "name_en": "Pressure at 700 hPa",
            "name_de": "Druck auf 700 hPa",
            "unit": "hPa",
        },
        "prestos": {
            "name_en": "Station pressure",
            "name_de": "Stationsdruck",
            "unit": "hPa",
        },
        # Humidity parameters
        "uty200": {
            "name_en": "Relative humidity 2m",
            "name_de": "Relative Luftfeuchtigkeit 2m",
            "unit": "%",
        },
        #'ure200': {'name_en': 'Relative humidity 2m', 'name_de': 'Relative Luftfeuchtigkeit 2m', 'unit': '%'},
        "pva200": {
            "name_en": "Water vapor pressure",
            "name_de": "Wasserdampfdruck",
            "unit": "hPa",
        },
        # Snow parameters
        "hto000": {"name_en": "Snow depth", "name_de": "Schneehöhe", "unit": "cm"},
        "htoaut": {
            "name_en": "Snow depth automatic",
            "name_de": "Schneehöhe automatisch",
            "unit": "cm",
        },
        "wcc006": {"name_en": "Weather code", "name_de": "Wettercode", "unit": ""},
        # Soil parameters
        "tso002": {
            "name_en": "Soil temperature 2cm",
            "name_de": "Bodentemperatur 2cm",
            "unit": "°C",
        },
        "tso005": {
            "name_en": "Soil temperature 5cm",
            "name_de": "Bodentemperatur 5cm",
            "unit": "°C",
        },
        "tso010": {
            "name_en": "Soil temperature 10cm",
            "name_de": "Bodentemperatur 10cm",
            "unit": "°C",
        },
        "tso020": {
            "name_en": "Soil temperature 20cm",
            "name_de": "Bodentemperatur 20cm",
            "unit": "°C",
        },
        "tso050": {
            "name_en": "Soil temperature 50cm",
            "name_de": "Bodentemperatur 50cm",
            "unit": "°C",
        },
        "tso100": {
            "name_en": "Soil temperature 100cm",
            "name_de": "Bodentemperatur 100cm",
            "unit": "°C",
        },
    }

    # Temporal suffix meanings
    SUFFIX_MEANINGS: ClassVar[dict[str, str]] = {
        "s0": "instant (10-min)",
        "h0": "hourly mean",
        "h1": "hourly vector/gust",
        "h3": "hourly 3-hour",
        "hn": "hourly min",
        "hx": "hourly max",
        "hs": "hourly sum",
        "d0": "daily mean",
        "dn": "daily min",
        "dx": "daily max",
        "m0": "monthly mean",
        "y0": "yearly mean",
        "z0": "10-min",
        "z1": "1-sec",
    }

    # Resolution mapping
    RESOLUTION_MAP: ClassVar[dict[str, str]] = {
        "t": "10min",
        "h": "hourly",
        "d": "daily",
        "m": "monthly",
        "y": "yearly",
    }

    def __init__(self, timeout: int = 30, language: str = "en"):
        """
        Initialize MeteoSwiss client.

        Args:
            timeout: Request timeout in seconds
            language: Language for column names ('en', 'de', 'fr', 'it')
        """
        self.timeout = timeout
        self.language = language
        self.session = requests.Session()
        self.parameter_metadata: dict[str, dict[str, str]] = {}

        self._load_parameter_metadata()

    def _load_parameter_metadata(self) -> None:
        """
        Download and cache parameter metadata from MeteoSwiss.

        This provides official parameter names, descriptions, and units.
        """
        try:
            logger.info("Downloading parameter metadata from MeteoSwiss...")
            response = self.session.get(
                self.PARAMETER_METADATA_URL, timeout=self.timeout
            )
            response.raise_for_status()

            metadata_df = pd.read_csv(
                StringIO(response.text), sep=";", encoding="windows-1252"
            )

            self.parameter_metadata = {}

            for _, row in metadata_df.iterrows():
                param_code = row.get("parameter_shortname", "")

                # Get description in selected language
                desc_col = f"Description_{self.language.upper()}"
                if desc_col not in metadata_df.columns:
                    desc_col = "Description_EN"  # Fallback to English

                description = row.get(desc_col, row.get("Description", ""))
                unit = row.get("Unit", "")

                if param_code:
                    self.parameter_metadata[param_code] = {
                        "name": description,
                        "unit": unit,
                    }

            logger.info(f"Loaded {len(self.parameter_metadata)} parameter definitions")

        except Exception as e:
            logger.warning(f"Could not load parameter metadata: {e}")
            logger.info("Using fallback parameter definitions")
            self.parameter_metadata = None

    def _parse_parameter_code(self, col: str) -> tuple:
        """
        Parse a parameter code into base and suffix.

        Examples:
            tre200h0 -> ('tre200', 'h0')
            rre150d0 -> ('rre150', 'd0')
            fu3010hn -> ('fu3010', 'hn')

        Returns:
            (base_code, suffix)
        """
        # Match pattern: letters+digits followed by temporal suffix
        match = re.match(r"^([a-z]+\d+)([a-z]\d|[a-z][nx])$", col.lower())
        if match:
            return match.group(1), match.group(2)

        # Try alternative pattern for codes without numbers at end
        match = re.match(r"^([a-z]+\d+)([a-z]+)$", col.lower())
        if match:
            base = match.group(1)
            suffix = match.group(2)
            # Only return if suffix is known temporal suffix
            if suffix in self.SUFFIX_MEANINGS or len(suffix) == 2:
                return base, suffix

        return col.lower(), ""

    def _get_column_rename_mapping(
        self, df: pd.DataFrame, include_units: bool = True
    ) -> dict[str, str]:
        """
        Create mapping from technical column names to meaningful names.

        Handles temporal suffixes (h0, d0, hn, hx, etc.)

        Args:
            df: DataFrame with technical column names
            include_units: Whether to include units in column names

        Returns:
            dictionary mapping old names to new names
        """
        rename_map = {}

        name_field = f"name_{self.language}"

        for col in df.columns:
            # Skip non-parameter columns
            if col.lower() in [
                "reference_timestamp",
                "station",
                "period",
                "resolution",
                "time",
                "station_abbr",
                "reference_timestamp",
            ]:
                continue

            # Parse parameter code into base and suffix
            base_code, suffix = self._parse_parameter_code(col)

            # Look up base parameter
            param_info = self.PARAMETER_DEFINITIONS.get(base_code, None)

            if param_info:
                # Get name in selected language
                name = param_info.get(name_field, param_info.get("name_en", col))
                unit = param_info.get("unit", "")

                # Add suffix meaning to name
                if suffix and suffix in self.SUFFIX_MEANINGS:
                    self.SUFFIX_MEANINGS[suffix]
                    # Only add suffix if it's not already implied in the name
                    if suffix in ["hn", "hx", "dn", "dx"]:
                        # For min/max suffixes
                        if "min" not in name.lower() and "max" not in name.lower():
                            aggregation = "min" if suffix.endswith("n") else "max"
                            name = f"{name} ({aggregation})"
                    elif suffix in ["h1", "z1"]:
                        # For vector/gust variants
                        if "vector" not in name.lower() and "gust" not in name.lower():
                            name = f"{name} (vector)"

                # Add unit
                if include_units and unit:
                    new_name = f"{name} [{unit}]"
                else:
                    new_name = name

                rename_map[col] = new_name
                logger.debug(f"Mapped: {col} -> {new_name}")
            else:
                # Keep original if no metadata found
                logger.debug(f"No metadata for parameter: {col} (base: {base_code})")
                rename_map[col] = col

        return rename_map

    def _query_stac_geojson(
        self, station_id: str | None = None, limit: int = 100
    ) -> list[dict]:
        """Query STAC API and parse GeoJSON response."""
        try:
            items_url = f"{self.STAC_API_URL}/collections/{self.COLLECTION_ID}/items"
            params = {"limit": limit}

            logger.info(f"Querying STAC API: {items_url}")
            response = self.session.get(items_url, params=params, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()
            features = data.get("features", [])
            logger.info(f"Found {len(features)} features in STAC response")

            if station_id:
                filtered = [
                    f for f in features if f.get("id", "").upper() == station_id.upper()
                ]
                logger.info(
                    f"Filtered to {len(filtered)} features for station {station_id}"
                )
                return filtered

            return features

        except Exception as e:
            logger.error(f"Failed to query STAC API: {e}", exc_info=True)
            return []

    def _extract_assets_from_features(
        self, features: list[dict], aggregation: str = "daily"
    ) -> list[dict]:
        """Extract CSV asset information from STAC features."""
        assets_info = []

        resolution_code = {
            "10min": "t",
            "hourly": "h",
            "daily": "d",
            "monthly": "m",
            "yearly": "y",
        }.get(aggregation, "d")

        for feature in features:
            station_id = feature.get("id", "")
            assets = feature.get("assets", {})

            for asset_name, asset_info in assets.items():
                if asset_info.get("type") != "text/csv":
                    continue

                parts = asset_name.replace(".csv", "").split("_")

                if len(parts) < 3:
                    continue

                file_resolution = parts[2] if len(parts) >= 3 else None
                period = "_".join(parts[3:]) if len(parts) >= 4 else "all"

                if file_resolution == resolution_code:
                    assets_info.append(
                        {
                            "station": station_id,
                            "asset_name": asset_name,
                            "url": asset_info.get("href", ""),
                            "resolution": self.RESOLUTION_MAP.get(
                                file_resolution, file_resolution
                            ),
                            "period": period,
                            "updated": asset_info.get("updated", ""),
                        }
                    )

        logger.info(
            f"Found {len(assets_info)} CSV assets matching resolution '{aggregation}'"
        )
        return assets_info

    def _download_and_parse_csv(
        self,
        url: str,
        station_id: str,
        start_date: pd.Timestamp | None = None,
        end_date: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Download and parse a MeteoSwiss CSV file with optional date filtering."""
        try:
            logger.info(f"Downloading CSV from: {url}")
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            df = pd.read_csv(
                StringIO(response.text),
                sep=";",
                encoding="windows-1252",
                na_values=["", " ", "-"],
            )

            if len(df) == 0:
                return None

            df = self._parse_meteoswiss_datetime(df)

            # Early date filtering
            if "reference_timestamp" in df.columns and (start_date or end_date):
                original_len = len(df)
                df["reference_timestamp"] = pd.to_datetime(
                    df["reference_timestamp"], errors="coerce"
                )
                if start_date:
                    df = df[df["reference_timestamp"] >= start_date]

                if end_date:
                    df = df[df["reference_timestamp"] <= end_date]

                logger.info(f"Date filtering: {original_len} -> {len(df)} rows")

                if len(df) == 0:
                    return None

            df["station"] = station_id
            return df

        except Exception as e:
            logger.error(f"Failed to download/parse CSV: {e}")
            return None

    def _parse_meteoswiss_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse MeteoSwiss datetime format (yyyymmddHHMM)."""
        time_col = None
        for col in df.columns:
            if col.lower() in ["time", "date", "datetime"]:
                time_col = col
                break

        if time_col is None:
            return df

        try:
            df[time_col] = df[time_col].astype(str).str.strip()
            df[time_col] = pd.to_datetime(
                df[time_col], format="%Y%m%d%H%M", errors="coerce"
            )
            df.rename(columns={time_col: "date"}, inplace=True)

            before = len(df)
            df = df.dropna(subset=["date"])
            after = len(df)

            if before != after:
                logger.info(f"Removed {before - after} rows with invalid dates")

        except Exception as e:
            logger.warning(f"Failed to parse datetime: {e}")

        return df

    def list_available_parameters(self) -> dict[str, dict]:
        """list all available weather parameters."""
        return (
            self.parameter_metadata
            if self.parameter_metadata
            else self.COMMON_PARAMETERS
        )

    def get_solothurn_stations(self) -> dict[str, Station]:
        """Get stations in Kanton Solothurn."""
        return self.HARDCODED_STATIONS.copy()

    def get_station_data(
        self,
        station_id: str,
        start_date: str | None = None,
        end_date: str | None = None,
        aggregation: str = "daily",
        parameters: list[str] | None = None,
        rename_columns: bool = True,
        include_units: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch station data from STAC API for specified datetime range.

        Args:
            station_id: Station identifier (e.g., 'GRE')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            aggregation: Time resolution ('10min', 'hourly', 'daily', 'monthly', 'yearly')
            parameters: Optional list of parameter IDs to include
            rename_columns: Whether to rename columns to meaningful names
            include_units: Whether to include units in column names

        Returns:
            DataFrame with weather data
        """
        logger.info(f"Fetching {station_id} data, aggregation: {aggregation}")

        try:
            start_dt = pd.to_datetime(start_date) if start_date else None
            end_dt = pd.to_datetime(end_date) if end_date else None

            # Step 1: Query STAC
            features = self._query_stac_geojson(station_id=station_id, limit=500)
            if not features:
                return None

            # Step 2: Extract CSV assets
            assets = self._extract_assets_from_features(features, aggregation)
            if not assets:
                return None

            # Step 3: Download and parse CSVs
            all_dataframes = []
            for asset in assets:
                df = self._download_and_parse_csv(
                    asset["url"], asset["station"], start_date=start_dt, end_date=end_dt
                )
                if df is not None and len(df) > 0:
                    df["period"] = asset["period"]
                    df["resolution"] = asset["resolution"]
                    all_dataframes.append(df)

            if not all_dataframes:
                return None

            # Step 4: Combine dataframes
            combined_df = pd.concat(all_dataframes, ignore_index=True)

            # Step 5: Remove duplicates
            if "reference_timestamp" in combined_df.columns:
                before = len(combined_df)
                combined_df = combined_df.drop_duplicates(
                    subset=["reference_timestamp", "station"], keep="first"
                )
                after = len(combined_df)
                if before != after:
                    logger.info(f"Removed {before - after} duplicate rows")

            if parameters:
                available_params = [
                    col for col in parameters if col in combined_df.columns
                ]
                cols_to_keep = [
                    "reference_timestamp",
                    "station",
                    *available_params,
                    "period",
                    "resolution",
                ]
                combined_df = combined_df[
                    [col for col in cols_to_keep if col in combined_df.columns]
                ]

            # Step 8: Rename columns to meaningful names
            if rename_columns:
                rename_map = self._get_column_rename_mapping(combined_df, include_units)
                combined_df = combined_df.rename(columns=rename_map)
                logger.info(f"Renamed {len(rename_map)} columns to meaningful names")

            logger.info(
                f"Final dataset: {len(combined_df)} records with {len(combined_df.columns)} columns"
            )
            return combined_df

        except Exception as e:
            logger.error(f"Failed to fetch data: {e}", exc_info=True)
            return None


class HistoricWeatherHandler:
    """
    High-level handler for historic MeteoSwiss weather data.

    Accepts external dependencies for flexibility:
      - stations_handler: SwissWeatherStations instance
      - params_loader: MetaParametersLoader instance
      - meteoswiss_client: MeteoSwissClient instance

    Typical usage:
        from metadata.stations import SwissWeatherStations
        from metadata.parameters import MetaParametersLoader
        from historic.client import MeteoSwissClient
        from historic_handler import HistoricWeatherHandler

        # Initialize dependencies
        stations = SwissWeatherStations()
        stations.load()
        params_loader = MetaParametersLoader()
        client = MeteoSwissClient()

        # Create handler with dependencies
        handler = HistoricWeatherHandler(
            stations_handler=stations,
            params_loader=params_loader,
            meteoswiss_client=client
        )

        # Fetch data
        result = handler.get_historic_by_name(
            station_name="Grenchen",
            start_date="2020-01-01",
            end_date="2020-12-31",
            parameters=["tre200d0", "rre150d0"]
        )
    """

    def __init__(
        self,
        stations_handler,
        params_loader,
        meteoswiss_client,
        language: str = "en",
    ):
        """
        Initialize the historic weather handler.

        Args:
            stations_handler: SwissWeatherStations instance (loaded)
            params_loader: MetaParametersLoader instance
            meteoswiss_client: MeteoSwissClient instance
            language: Language for output ('en', 'de', 'fr', 'it')

        Raises:
            TypeError: If dependencies are None or wrong type
        """
        if stations_handler is None:
            raise TypeError("stations_handler cannot be None")
        if params_loader is None:
            raise TypeError("params_loader cannot be None")
        if meteoswiss_client is None:
            raise TypeError("meteoswiss_client cannot be None")

        self.stations = stations_handler
        self.params_loader = params_loader
        self.client = meteoswiss_client
        self.language = language
        self._parameters_source_name = "smn_parameters"

        logger.info("HistoricWeatherHandler initialized")

    def _resolve_station_by_name(
        self,
        station_name: str,
        exact: bool = True,
    ):
        """
        Resolve a station from SwissWeatherStations by name.

        Args:
            station_name: Station name (e.g., 'Grenchen')
            exact: If True, require exact match

        Returns:
            WeatherStation object

        Raises:
            ValueError: If not found or ambiguous
        """
        candidates = self.stations.get_by_name(station_name, exact=exact)

        if not candidates:
            raise ValueError(f"No station found for name '{station_name}'")

        if len(candidates) > 1 and not exact:
            # Heuristic: prefer exact case-insensitive match
            exact_ci = [
                s for s in candidates if s.point_name.lower() == station_name.lower()
            ]
            if len(exact_ci) == 1:
                return exact_ci[0]
            raise ValueError(
                f"Ambiguous station name '{station_name}', "
                f"found {len(candidates)} matches; use exact=True or ID/abbr."
            )

        return candidates[0]

    def _resolve_station_by_coords(
        self,
        lat: float,
        lon: float,
        n: int = 1,
    ):
        """
        Resolve nearest station to coordinates.

        Args:
            lat: Latitude
            lon: Longitude
            n: Number of nearest stations to return

        Returns:
            (WeatherStation, distance_km)

        Raises:
            ValueError: If no stations found
        """
        nearest = self.stations.find_nearest(lat, lon, n=n)
        if not nearest:
            raise ValueError("No stations found for given coordinates")
        return nearest[0]

    def _resolve_station_id_for_historic(self, sws_station) -> str:
        """
        Map SwissWeatherStations station to SMN station ID.

        Strategy:
          1. If abbr exists in client HARDCODED_STATIONS -> use it
          2. Otherwise use abbreviation directly

        Args:
            sws_station: WeatherStation object

        Returns:
            SMN station ID (e.g., 'GRE', 'BER')
        """
        abbr = sws_station.station_abbr
        if hasattr(self.client, "HARDCODED_STATIONS"):
            if abbr in self.client.HARDCODED_STATIONS:
                return abbr
        return abbr

    def list_available_parameters(self) -> dict[str, dict[str, Any]]:
        """
        list all available parameters for historic data.

        Returns:
            dict: param_code -> metadata dict with name/unit/etc.
        """
        # Get from client
        client_params = self.client.list_available_parameters()

        # Try to get from params_loader
        try:
            extra_meta = self.params_loader.get_all(source=self._parameters_source_name)
        except Exception:
            extra_meta = {}

        merged: dict[str, dict[str, Any]] = {}

        # Start with client info
        for code, meta in client_params.items():
            merged[code] = dict(meta)

        # Enrich with loader metadata
        for key, param_meta in extra_meta.items():
            shortname = param_meta.get("parameter_shortname") or key
            if shortname not in merged:
                merged[shortname] = {}
            if hasattr(param_meta, "data"):
                merged[shortname].update(param_meta.data)
            else:
                merged[shortname].update(param_meta)

        return merged

    def get_historic_by_station_id(
        self,
        station_id: str,
        start_date: str | None = None,
        end_date: str | None = None,
        aggregation: str = "daily",
        parameters: list[str] | None = None,
        rename_columns: bool = False,
        include_units: bool = False,
    ) -> HistoricQueryResult | None:
        """
        Fetch historic data by SMN station ID.

        Args:
            station_id: SMN station ID (e.g., 'GRE', 'BER')
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            aggregation: Time resolution ('10min', 'hourly', 'daily', 'monthly', 'yearly')
            parameters: list of parameter codes
            rename_columns: Rename to meaningful names
            include_units: Include units in column names

        Returns:
            HistoricQueryResult or None

        Example:
            result = handler.get_historic_by_station_id(
                station_id='GRE',
                start_date='2020-01-01',
                end_date='2020-12-31',
                aggregation='daily',
                parameters=['tre200d0', 'rre150d0']
            )
        """
        logger.info(f"Fetching historic data for station {station_id}")

        # Fetch from client
        df = self.client.get_station_data(
            station_id=station_id,
            start_date=start_date,
            end_date=end_date,
            aggregation=aggregation,
            parameters=parameters,
            rename_columns=rename_columns,
            include_units=include_units,
        )

        if df is None or df.empty:
            logger.warning(f"No data for station {station_id}")
            return None

        # Get station metadata
        station_meta = self._get_station_metadata_for_id(station_id)

        # Parse dates
        start_ts = pd.to_datetime(start_date) if start_date else None
        end_ts = pd.to_datetime(end_date) if end_date else None
        used_params = parameters or [
            c
            for c in df.columns
            if c not in {"date", "station", "period", "resolution"}
        ]

        result = HistoricQueryResult(
            station_id=station_id,
            station_name=station_meta["name"],
            canton=station_meta.get("canton"),
            latitude=station_meta["latitude"],
            longitude=station_meta["longitude"],
            altitude=station_meta["altitude"],
            aggregation=aggregation,
            parameters=used_params,
            start_date=start_ts,
            end_date=end_ts,
            data=df,
        )

        logger.info(f"✓ Retrieved {len(df)} records for {station_meta['name']}")
        return result

    def get_historic_by_name(
        self,
        station_name: str,
        start_date: str | None = None,
        end_date: str | None = None,
        aggregation: str = "daily",
        parameters: list[str] | None = None,
        exact: bool = True,
        rename_columns: bool = True,
        include_units: bool = True,
    ) -> HistoricQueryResult | None:
        """
        Fetch historic data by Swiss station name.

        Args:
            station_name: Swiss station name (e.g., 'Grenchen')
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            aggregation: Time resolution
            parameters: list of parameter codes
            exact: Require exact station name match
            rename_columns: Rename to meaningful names
            include_units: Include units in column names

        Returns:
            HistoricQueryResult or None

        Raises:
            ValueError: If station not found or ambiguous

        Example:
            result = handler.get_historic_by_name(
                station_name='Grenchen',
                start_date='2020-01-01',
                end_date='2020-12-31',
                parameters=['tre200d0']
            )
        """
        logger.info(f"Resolving station by name: '{station_name}'")

        # Resolve station from SwissWeatherStations
        sws_station = self._resolve_station_by_name(station_name, exact=exact)
        station_id = self._resolve_station_id_for_historic(sws_station)

        logger.info(f"Resolved to station ID: {station_id}")

        return self.get_historic_by_station_id(
            station_id=station_id,
            start_date=start_date,
            end_date=end_date,
            aggregation=aggregation,
            parameters=parameters,
            rename_columns=rename_columns,
            include_units=include_units,
        )

    def get_historic_by_coords(
        self,
        lat: float,
        lon: float,
        start_date: str | None = None,
        end_date: str | None = None,
        aggregation: str = "daily",
        parameters: list[str] | None = None,
        rename_columns: bool = True,
        include_units: bool = True,
    ) -> HistoricQueryResult | None:
        """
        Fetch historic data for nearest station to coordinates.

        Args:
            lat: Latitude
            lon: Longitude
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            aggregation: Time resolution
            parameters: list of parameter codes
            rename_columns: Rename to meaningful names
            include_units: Include units in column names

        Returns:
            HistoricQueryResult for nearest station

        Example:
            result = handler.get_historic_by_coords(
                lat=47.21,
                lon=7.54,
                start_date='2020-01-01',
                end_date='2020-12-31'
            )
        """
        logger.info(f"Finding nearest station to {lat}, {lon}")

        # Find nearest station
        sws_station, distance = self._resolve_station_by_coords(lat, lon, n=1)
        station_id = self._resolve_station_id_for_historic(sws_station)

        logger.info(
            f"Nearest station: {sws_station.point_name} ({distance:.1f}km away)"
        )

        return self.get_historic_by_station_id(
            station_id=station_id,
            start_date=start_date,
            end_date=end_date,
            aggregation=aggregation,
            parameters=parameters,
            rename_columns=rename_columns,
            include_units=include_units,
        )

    def get_temperature_history(
        self,
        station_name_or_id: str,
        start_date: str,
        end_date: str,
        daily: bool = True,
    ) -> pd.DataFrame:
        """
        Get temperature history for a station.

        Args:
            station_name_or_id: Station name or SMN ID
            start_date: 'YYYY-MM-DD'
            end_date: 'YYYY-MM-DD'
            daily: Use daily aggregation if True

        Returns:
            DataFrame indexed by date
        """
        # Detect if ID or name
        if station_name_or_id.isupper() and len(station_name_or_id) <= 4:
            # Likely ID
            result = self.get_historic_by_station_id(
                station_id=station_name_or_id,
                start_date=start_date,
                end_date=end_date,
                aggregation="daily" if daily else "10min",
                parameters=["tre200d0", "tre200dn", "tre200dx"]
                if daily
                else ["tre200s0"],
            )
        else:
            # Likely name
            result = self.get_historic_by_name(
                station_name=station_name_or_id,
                start_date=start_date,
                end_date=end_date,
                aggregation="daily" if daily else "10min",
                parameters=["tre200d0", "tre200dn", "tre200dx"]
                if daily
                else ["tre200s0"],
            )

        if not result:
            return None

        df = result.data.copy()
        if "date" in df.columns:
            df = df.set_index("date").sort_index()
        return df

    def get_precipitation_history(
        self,
        station_name_or_id: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Get precipitation history for a station.

        Args:
            station_name_or_id: Station name or SMN ID
            start_date: 'YYYY-MM-DD'
            end_date: 'YYYY-MM-DD'

        Returns:
            DataFrame indexed by date
        """
        if station_name_or_id.isupper() and len(station_name_or_id) <= 4:
            result = self.get_historic_by_station_id(
                station_id=station_name_or_id,
                start_date=start_date,
                end_date=end_date,
                aggregation="daily",
                parameters=["rre150d0"],
            )
        else:
            result = self.get_historic_by_name(
                station_name=station_name_or_id,
                start_date=start_date,
                end_date=end_date,
                aggregation="daily",
                parameters=["rre150d0"],
            )

        if not result:
            return None

        df = result.data.copy()
        if "date" in df.columns:
            df = df.set_index("date").sort_index()
        return df

    def _get_station_metadata_for_id(self, station_id: str) -> dict[str, Any]:
        """
        Get station metadata for an SMN station ID.

        Args:
            station_id: SMN station ID

        Returns:
            dict with name, location, elevation
        """
        if hasattr(self.client, "HARDCODED_STATIONS"):
            if station_id in self.client.HARDCODED_STATIONS:
                station = self.client.HARDCODED_STATIONS[station_id]
                return {
                    "name": station.name,
                    "canton": station.canton,
                    "latitude": station.latitude,
                    "longitude": station.longitude,
                    "altitude": station.altitude,
                }

        # Try SwissWeatherStations by abbreviation
        try:
            sws_station = self.stations.get_by_abbr(station_id)
            if sws_station:
                return {
                    "name": sws_station.point_name,
                    "canton": getattr(sws_station, "canton", ""),
                    "latitude": sws_station.coordinates_wgs84_lat,
                    "longitude": sws_station.coordinates_wgs84_lon,
                    "altitude": sws_station.point_height_masl,
                }
        except Exception:
            pass

        # Fallback
        return {
            "name": station_id,
            "canton": "",
            "latitude": float("nan"),
            "longitude": float("nan"),
            "altitude": -1,
        }

    def export_to_csv(
        self,
        result: HistoricQueryResult,
        filepath: str,
        encoding: str = "utf-8",
    ) -> bool:
        """Export historic data to CSV."""
        try:
            result.data.to_csv(filepath, index=False, encoding=encoding, sep=",")
            logger.info(f"✓ Exported to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to export to CSV: {e}")
            return False

    def export_to_json(
        self,
        result: HistoricQueryResult,
        filepath: str,
    ) -> bool:
        """Export historic data to JSON."""
        try:
            result.data.to_json(filepath, orient="records", indent=2)
            logger.info(f"✓ Exported to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to export to JSON: {e}")
            return False

from __future__ import annotations

import math
from typing import Any
from dataclasses import dataclass
from datetime import datetime
from io import StringIO

import pandas as pd
import requests


@dataclass
class WeatherStation:
    """Weather station information"""

    point_id: int
    point_type_id: int
    station_abbr: str
    postal_code: int | None
    point_name: str
    point_type: str
    point_height_masl: float
    coordinates_lv95_east: float
    coordinates_lv95_north: float
    coordinates_wgs84_lat: float
    coordinates_wgs84_lon: float

    @classmethod
    def from_series(cls, row: pd.Series) -> WeatherStation:
        """Create from pandas Series"""
        return cls(
            point_id=int(row["point_id"]),
            point_type_id=int(row["point_type_id"]),
            station_abbr=str(row["station_abbr"]),
            postal_code=int(row["postal_code"])
            if pd.notna(row["postal_code"])
            else None,
            point_name=str(row["point_name"]),
            point_type=str(row["point_type_en"]),
            point_height_masl=float(row["point_height_masl"]),
            coordinates_lv95_east=float(row["point_coordinates_lv95_east"]),
            coordinates_lv95_north=float(row["point_coordinates_lv95_north"]),
            coordinates_wgs84_lat=float(row["point_coordinates_wgs84_lat"]),
            coordinates_wgs84_lon=float(row["point_coordinates_wgs84_lon"]),
        )

    @property
    def coordinates(self) -> tuple[float, float]:
        """Get (lat, lon) coordinates"""
        return (self.coordinates_wgs84_lat, self.coordinates_wgs84_lon)

    @property
    def elevation(self) -> float:
        """Get elevation in meters"""
        return self.point_height_masl

    def __repr__(self):
        return f"WeatherStation({self.station_abbr}: {self.point_name}, {self.elevation:.0f}m)"


class SwissWeatherStations:
    """
    Swiss Weather Stations Handler

    Manages all MeteoSwiss weather stations with:
    - Search by name, ID, abbreviation
    - Geographic queries
    - Elevation filtering
    - Region-based selection
    - Distance calculations

    Example:
        # Initialize
        stations = SwissWeatherStations()
        stations.load()

        # Get station by abbreviation
        gre = stations.get_by_abbr('GRE')
        print(f"{gre.point_name}: {gre.elevation}m")

        # Find nearby stations
        nearby = stations.find_nearby(47.18, 7.42, radius_km=20)
        print(f"Found {len(nearby)} stations within 20km")

        # Search by name
        zurich_stations = stations.search('Zürich')
    """

    STATIONS_URL = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-local-forecasting/ogd-local-forcasting_meta_point.csv"

    def __init__(self):
        """Initialize handler"""
        self.df: pd.DataFrame = None
        self._loaded: bool = False
        self._cache_timestamp: datetime | None = None

    def load(self, force_reload: bool = False, timeout: int = 60) -> pd.DataFrame:
        """
        Load stations from MeteoSwiss API

        Args:
            force_reload: Force reload even if cached
            timeout: Request timeout in seconds

        Returns:
            DataFrame with station data

        Example:
            stations = SwissWeatherStations()
            df = stations.load()
            print(f"Loaded {len(df)} stations")
        """
        if self._loaded and not force_reload:
            return self.df

        print("Loading Swiss weather stations from MeteoSwiss...")

        # Download CSV
        response = requests.get(self.STATIONS_URL, timeout=timeout)
        response.raise_for_status()

        # Parse with Latin-1 encoding and semicolon delimiter
        content = response.content.decode("latin-1")
        self.df = pd.read_csv(StringIO(content), sep=";")

        self._loaded = True
        self._cache_timestamp = datetime.now()

        print(f"✓ Loaded {len(self.df)} weather stations")

        return self.df

    def reload(self, timeout: int = 60) -> pd.DataFrame:
        """Force reload stations data"""
        return self.load(force_reload=True, timeout=timeout)

    def get_by_id(self, point_id: int) -> WeatherStation | None:
        """
        Get station by point ID

        Args:
            point_id: Station point ID

        Returns:
            WeatherStation or None

        Example:
            gre = stations.get_by_id(774)
            print(gre.point_name)  # "Grenchen"
        """
        self._ensure_loaded()

        result = self.df[self.df["point_id"] == point_id]
        if len(result) == 0:
            return None

        return WeatherStation.from_series(result.iloc[0])

    def get_by_abbr(self, abbreviation: str) -> WeatherStation | None:
        """
        Get station by abbreviation

        Args:
            abbreviation: Station abbreviation (e.g., 'GRE', 'ZRH')

        Returns:
            WeatherStation or None

        Example:
            gre = stations.get_by_abbr('GRE')
            print(gre.point_name)  # "Grenchen"
        """
        self._ensure_loaded()

        result = self.df[self.df["station_abbr"] == abbreviation.upper()]
        if len(result) == 0:
            return None

        return WeatherStation.from_series(result.iloc[0])

    def get_by_name(self, name: str, exact: bool = False) -> list[WeatherStation]:
        """
        Get stations by name

        Args:
            name: Station name (case-insensitive)
            exact: Require exact match

        Returns:
            list of matching stations

        Example:
            # Partial match
            zurich = stations.get_by_name('Zürich')

            # Exact match
            zurich = stations.get_by_name('Zürich / Fluntern', exact=True)
        """
        self._ensure_loaded()

        if exact:
            results = self.df[self.df["point_name"].str.lower() == name.lower()]
        else:
            results = self.df[
                self.df["point_name"].str.contains(name, case=False, na=False)
            ]

        return [WeatherStation.from_series(row) for _, row in results.iterrows()]

    def search(
        self, keyword: str, search_in: list[str] | None = None
    ) -> list[WeatherStation]:
        """
        Search stations by keyword

        Args:
            keyword: Search term
            search_in: Columns to search in (default: name and abbreviation)

        Returns:
            list of matching stations

        Example:
            # Search in name
            results = stations.search('Basel')

            # Search in multiple fields
            results = stations.search('4500', search_in=['postal_code'])
        """
        self._ensure_loaded()

        if search_in is None:
            search_in = ["point_name", "station_abbr"]

        keyword_lower = keyword.lower()
        mask = pd.Series([False] * len(self.df))

        for col in search_in:
            if col in self.df.columns:
                mask |= (
                    self.df[col]
                    .astype(str)
                    .str.lower()
                    .str.contains(keyword_lower, na=False)
                )

        results = self.df[mask]
        return [WeatherStation.from_series(row) for _, row in results.iterrows()]

    def find_nearby(
        self,
        lat: float,
        lon: float,
        radius_km: float = 10.0,
        max_results: int | None = None,
    ) -> list[tuple[WeatherStation, float]]:
        """
        Find stations within radius of coordinates

        Args:
            lat: Latitude
            lon: Longitude
            radius_km: Search radius in kilometers
            max_results: Maximum number of results

        Returns:
            list of (station, distance_km) tuples, sorted by distance

        Example:
            # Find stations within 20km of Bern
            nearby = stations.find_nearby(46.95, 7.45, radius_km=20)
            for station, distance in nearby:
                print(f"{station.point_name}: {distance:.1f}km")
        """
        self._ensure_loaded()

        results = []

        for _, row in self.df.iterrows():
            station = WeatherStation.from_series(row)
            distance = self._haversine_distance(
                lat, lon, station.coordinates_wgs84_lat, station.coordinates_wgs84_lon
            )

            if distance <= radius_km:
                results.append((station, distance))

        # Sort by distance
        results.sort(key=lambda x: x[1])

        if max_results:
            results = results[:max_results]

        return results

    def find_nearest(
        self, lat: float, lon: float, n: int = 1
    ) -> list[tuple[WeatherStation, float]]:
        """
        Find n nearest stations to coordinates

        Args:
            lat: Latitude
            lon: Longitude
            n: Number of stations to return

        Returns:
            list of (station, distance_km) tuples

        Example:
            # Find 5 nearest stations to Solothurn
            nearest = stations.find_nearest(47.21, 7.54, n=5)
            for station, distance in nearest:
                print(f"{station.point_name}: {distance:.1f}km")
        """
        self._ensure_loaded()

        distances = []

        for _, row in self.df.iterrows():
            station = WeatherStation.from_series(row)
            distance = self._haversine_distance(
                lat, lon, station.coordinates_wgs84_lat, station.coordinates_wgs84_lon
            )
            distances.append((station, distance))

        # Sort by distance and return top n
        distances.sort(key=lambda x: x[1])
        return distances[:n]

    def filter_by_bbox(
        self, lat_min: float, lat_max: float, lon_min: float, lon_max: float
    ) -> list[WeatherStation]:
        """
        Filter stations by bounding box

        Args:
            lat_min: Minimum latitude
            lat_max: Maximum latitude
            lon_min: Minimum longitude
            lon_max: Maximum longitude

        Returns:
            list of stations within bounding box

        Example:
            # Solothurn region
            solothurn = stations.filter_by_bbox(
                lat_min=47.0, lat_max=47.5,
                lon_min=7.3, lon_max=8.0
            )
            print(f"Found {len(solothurn)} stations in Solothurn")
        """
        self._ensure_loaded()

        mask = (
            (self.df["point_coordinates_wgs84_lat"] >= lat_min)
            & (self.df["point_coordinates_wgs84_lat"] <= lat_max)
            & (self.df["point_coordinates_wgs84_lon"] >= lon_min)
            & (self.df["point_coordinates_wgs84_lon"] <= lon_max)
        )

        results = self.df[mask]
        return [WeatherStation.from_series(row) for _, row in results.iterrows()]

    def filter_by_elevation(
        self,
        min_elevation: float | None = None,
        max_elevation: float | None = None,
    ) -> list[WeatherStation]:
        """
        Filter stations by elevation

        Args:
            min_elevation: Minimum elevation in meters
            max_elevation: Maximum elevation in meters

        Returns:
            list of stations within elevation range

        Example:
            # Alpine stations (> 1500m)
            alpine = stations.filter_by_elevation(min_elevation=1500)

            # Valley stations (< 500m)
            valley = stations.filter_by_elevation(max_elevation=500)
        """
        self._ensure_loaded()

        mask = pd.Series([True] * len(self.df))

        if min_elevation is not None:
            mask &= self.df["point_height_masl"] >= min_elevation

        if max_elevation is not None:
            mask &= self.df["point_height_masl"] <= max_elevation

        results = self.df[mask]
        return [WeatherStation.from_series(row) for _, row in results.iterrows()]

    def get_highest_stations(self, n: int = 10) -> list[WeatherStation]:
        """
        Get n highest elevation stations

        Example:
            top_10 = stations.get_highest_stations(10)
            for station in top_10:
                print(f"{station.point_name}: {station.elevation:.0f}m")
        """
        self._ensure_loaded()

        sorted_df = self.df.sort_values("point_height_masl", ascending=False)
        return [
            WeatherStation.from_series(row) for _, row in sorted_df.head(n).iterrows()
        ]

    def get_lowest_stations(self, n: int = 10) -> list[WeatherStation]:
        """Get n lowest elevation stations"""
        self._ensure_loaded()

        sorted_df = self.df.sort_values("point_height_masl", ascending=True)
        return [
            WeatherStation.from_series(row) for _, row in sorted_df.head(n).iterrows()
        ]

    def filter_by_type(self, station_type: str) -> list[WeatherStation]:
        """
        Filter stations by type

        Args:
            station_type: Station type (e.g., 'Station', 'Grid point')

        Returns:
            list of matching stations
        """
        self._ensure_loaded()

        results = self.df[self.df["point_type_en"] == station_type]
        return [WeatherStation.from_series(row) for _, row in results.iterrows()]

    def get_station_types(self) -> list[str]:
        """Get all unique station types"""
        self._ensure_loaded()
        return self.df["point_type_en"].unique().tolist()

    def filter_by_postal_code(self, postal_code: int) -> list[WeatherStation]:
        """Filter stations by postal code"""
        self._ensure_loaded()

        results = self.df[self.df["postal_code"] == postal_code]
        return [WeatherStation.from_series(row) for _, row in results.iterrows()]

    def get_statistics(self) -> dict:
        """
        Get statistics about stations

        Returns:
            dictionary with statistics
        """
        self._ensure_loaded()

        return {
            "total_stations": len(self.df),
            "station_types": self.df["point_type_en"].value_counts().to_dict(),
            "elevation_min": float(self.df["point_height_masl"].min()),
            "elevation_max": float(self.df["point_height_masl"].max()),
            "elevation_mean": float(self.df["point_height_masl"].mean()),
            "lat_min": float(self.df["point_coordinates_wgs84_lat"].min()),
            "lat_max": float(self.df["point_coordinates_wgs84_lat"].max()),
            "lon_min": float(self.df["point_coordinates_wgs84_lon"].min()),
            "lon_max": float(self.df["point_coordinates_wgs84_lon"].max()),
            "with_postal_code": int(self.df["postal_code"].notna().sum()),
        }

    def get_all_stations(self, 
                        limit: int = 50,
                        offset: int = 0,
                        sort_by: str = "name",
                        order: str = "asc",) -> dict[str, Any]:
        """Get all stations as list"""
        self._ensure_loaded()
        sort_key = (sort_by or "name").lower()
        order_key = (order or "asc").lower()

        sort_map = {
            "name": "point_name",
            "elevation": "point_height_masl",
            "abbr": "station_abbr",
            "id": "point_id",
            "type": "point_type_en",
            "postal_code": "postal_code",
            "lat": "point_coordinates_wgs84_lat",
            "lon": "point_coordinates_wgs84_lon",
        }

        if sort_key not in sort_map:
            raise ValueError(
                f"Invalid sort_by='{sort_by}'. Allowed: {', '.join(sorted(sort_map.keys()))}"
            )

        if order_key not in {"asc", "desc"}:
            raise ValueError("Invalid order. Allowed: asc, desc")

        df = self.df
        total = int(len(df))

        sort_col = sort_map[sort_key]
        ascending = (order_key == "asc")

        # Stable sort helps deterministic pagination
        df_sorted = df.sort_values(by=sort_col, ascending=ascending, kind="mergesort")

        start = int(offset)
        end = start + int(limit)
        df_page = df_sorted.iloc[start:end] if start < total else df_sorted.iloc[0:0]

        data = [WeatherStation.from_series(row) for _, row in df_page.iterrows()]

        return {
            "total": total,
            "limit": int(limit),
            "offset": int(offset),
            "data": data,
        }
    
    def count(self) -> int:
        """Get total number of stations"""
        self._ensure_loaded()
        return len(self.df)

    def to_dataframe(self) -> pd.DataFrame:
        """Get stations as DataFrame"""
        self._ensure_loaded()
        return self.df.copy()

    def export_to_csv(
        self,
        filepath: str,
        stations: list[WeatherStation] | None = None,
        encoding: str = "utf-8",
    ) -> None:
        """
        Export stations to CSV

        Args:
            filepath: Output file path
            stations: Stations to export (None = all)
            encoding: File encoding
        """
        if stations is None:
            df = self.df
        else:
            # Convert stations back to DataFrame
            data = [vars(s) for s in stations]
            df = pd.DataFrame(data)

        df.to_csv(filepath, index=False, encoding=encoding)

    def export_to_geojson(
        self, filepath: str, stations: list[WeatherStation] | None = None
    ) -> None:
        """
        Export stations to GeoJSON

        Args:
            filepath: Output file path
            stations: Stations to export (None = all)
        """
        if stations is None:
            stations = self.get_all_stations()

        features = []
        for station in stations:
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [
                        station.coordinates_wgs84_lon,
                        station.coordinates_wgs84_lat,
                    ],
                },
                "properties": {
                    "point_id": station.point_id,
                    "name": station.point_name,
                    "abbreviation": station.station_abbr,
                    "elevation": station.elevation,
                    "type": station.point_type,
                    "postal_code": station.postal_code,
                },
            }
            features.append(feature)

        geojson = {"type": "FeatureCollection", "features": features}

        import json

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(geojson, f, indent=2)

    def _ensure_loaded(self):
        """Ensure data is loaded"""
        if not self._loaded:
            self.load()

    @staticmethod
    def _haversine_distance(
        lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """
        Calculate distance between two points using Haversine formula

        Returns:
            Distance in kilometers
        """
        r = 6371  # Earth radius in km

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return r * c

    def summary(self) -> str:
        """Get handler summary"""
        self._ensure_loaded()

        stats = self.get_statistics()

        return f"""
Swiss Weather Stations Summary
===============================
Total Stations: {stats["total_stations"]:,}
Loaded: {self._cache_timestamp.strftime("%Y-%m-%d %H:%M:%S") if self._cache_timestamp else "Not loaded"}

Station Types:
{chr(10).join(f"  • {k}: {v:,}" for k, v in stats["station_types"].items())}

Elevation:
  Minimum: {stats["elevation_min"]:.0f}m
  Maximum: {stats["elevation_max"]:.0f}m
  Average: {stats["elevation_mean"]:.0f}m

Coverage:
  Latitude: {stats["lat_min"]:.2f}° to {stats["lat_max"]:.2f}°
  Longitude: {stats["lon_min"]:.2f}° to {stats["lon_max"]:.2f}°
  With Postal Code: {stats["with_postal_code"]:,}
"""

    def __repr__(self):
        return f"SwissWeatherStations(loaded={self._loaded}, count={len(self.df) if self._loaded else 0})"

    def __str__(self):
        return self.summary()

    def __len__(self):
        return self.count() if self._loaded else 0


if __name__ == "__main__":
    print("=" * 70)
    print("Swiss Weather Stations Handler - Usage Examples")
    print("=" * 70)

    # Initialize and load
    stations = SwissWeatherStations()
    stations.load()

    # Example 1: Get station by abbreviation
    print("\n[Example 1] Get Station by Abbreviation")
    print("-" * 70)
    gre = stations.get_by_abbr("GRE")
    if gre:
        print(f"✓ Found: {gre.point_name}")
        print(f"  ID: {gre.point_id}")
        print(f"  Elevation: {gre.elevation:.0f}m")
        print(f"  Coordinates: {gre.coordinates[0]:.6f}°N, {gre.coordinates[1]:.6f}°E")

    # Example 2: Search by name
    print("\n[Example 2] Search by Name")
    print("-" * 70)
    zurich = stations.get_by_name("Zürich")
    print(f"✓ Found {len(zurich)} stations with 'Zürich' in name:")
    for s in zurich[:5]:
        print(f"  • {s.point_name} ({s.station_abbr})")

    # Example 3: Find nearby stations
    print("\n[Example 3] Find Nearby Stations")
    print("-" * 70)
    nearby = stations.find_nearby(47.18, 7.42, radius_km=20)
    print(f"✓ Found {len(nearby)} stations within 20km of Grenchen:")
    for station, distance in nearby[:5]:
        print(f"  • {station.point_name:20s} - {distance:5.1f}km")

    # Example 4: Filter by region
    print("\n[Example 4] Filter by Bounding Box (Solothurn)")
    print("-" * 70)
    solothurn = stations.filter_by_bbox(47.0, 47.5, 7.3, 8.0)
    print(f"✓ Found {len(solothurn)} stations in Solothurn region")

    # Example 5: Elevation queries
    print("\n[Example 5] Elevation Queries")
    print("-" * 70)
    alpine = stations.filter_by_elevation(min_elevation=1500)
    print(f"✓ Alpine stations (>1500m): {len(alpine)}")

    highest = stations.get_highest_stations(5)
    print("\n  Top 5 highest:")
    for s in highest:
        print(f"    • {s.point_name:30s} {s.elevation:5.0f}m")

    # Example 6: Statistics
    print("\n[Example 6] Statistics")
    print("-" * 70)
    print(stations.summary())

    print("\n" + "=" * 70)
    print("✓ All examples completed!")
    print("=" * 70)

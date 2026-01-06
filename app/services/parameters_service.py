from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from typing import Any

import pandas as pd
import requests


@dataclass
class CSVSource:
    """CSV source configuration"""

    name: str
    url: str
    description: str | None = None
    encoding: str = "latin-1"
    delimiter: str = ";"
    key_column: str | None = None

    def __repr__(self):
        return f"CSVSource(name='{self.name}', url='{self.url[:50]}...')"


@dataclass
class ParameterMetadata:
    """Generic parameter metadata"""

    source: str
    data: dict[str | Hashable, Any]

    def get(self, key: str, default: Any = None) -> Any:
        """Get parameter value"""
        return self.data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """dictionary-style access"""
        return self.data[key]

    def __repr__(self):
        return f"ParameterMetadata(source='{self.source}', fields={len(self.data)})"


class MetaParametersLoader:
    """
    Generic CSV metadata parameters loader

    Loads and manages metadata from one or more CSV URLs with:
    - Automatic encoding detection
    - Delimiter handling
    - Caching
    - Search and filter
    - Export capabilities

    Example:
        # Initialize
        loader = MetaParametersLoader()

        # Add CSV source
        loader.add_source(
            name='parameters',
            url='https://example.com/parameters.csv',
            encoding='latin-1',
            delimiter=';',
            key_column='parameter_shortname'
        )

        # Load data
        loader.load_all()

        # Get parameter
        param = loader.get('tre200h0')
        print(param['parameter_description_en'])

        # Search
        results = loader.search('temperature')
    """

    def __init__(self) -> None:
        """Initialize loader"""
        self.sources: dict[str, CSVSource] = {}
        self.data: dict[str, pd.DataFrame] = {}
        self.metadata: dict[str, ParameterMetadata] = {}
        self._cache_timestamp: dict[str, datetime] = {}

    def add_source(
        self,
        name: str,
        url: str,
        description: str | None = None,
        encoding: str = "latin-1",
        delimiter: str = ";",
        key_column: str | None = None,
    ) -> None:
        """
        Add CSV source

        Args:
            name: Source name (unique identifier)
            url: CSV file URL
            description: Source description
            encoding: Text encoding (default: 'latin-1')
            delimiter: CSV delimiter (default: ';')
            key_column: Column to use as key for lookups

        Example:
            loader.add_source(
                name='parameters',
                url='https://data.geo.admin.ch/.../parameters.csv',
                encoding='latin-1',
                delimiter=';',
                key_column='parameter_shortname'
            )
        """
        source = CSVSource(
            name=name,
            url=url,
            description=description,
            encoding=encoding,
            delimiter=delimiter,
            key_column=key_column,
        )
        self.sources[name] = source

    def remove_source(self, name: str) -> bool:
        """Remove CSV source"""
        if name in self.sources:
            del self.sources[name]
            if name in self.data:
                del self.data[name]
            return True
        return False

    def list_sources(self) -> list[str]:
        """list all source names"""
        return list(self.sources.keys())

    def get_source(self, name: str) -> CSVSource | None:
        """Get source configuration"""
        return self.sources.get(name)

    def load_source(
        self, name: str, force_reload: bool = False, timeout: int = 60
    ) -> pd.DataFrame:
        """
        Load data from specific source

        Args:
            name: Source name
            force_reload: Force reload even if cached
            timeout: Request timeout in seconds

        Returns:
            DataFrame with loaded data

        Example:
            df = loader.load_source('parameters')
            print(df.head())
        """
        # Check cache
        if name in self.data and not force_reload:
            return self.data[name]

        # Get source
        source = self.sources.get(name)
        if not source:
            raise ValueError(f"Source '{name}' not found")

        # Download CSV
        print(f"Loading {name} from {source.url}...")
        response = requests.get(source.url, timeout=timeout)
        response.raise_for_status()

        # Decode with specified encoding
        try:
            content = response.content.decode(source.encoding)
        except UnicodeDecodeError:
            # Try alternative encodings
            print(f"  Warning: {source.encoding} failed, trying utf-8...")
            try:
                content = response.content.decode("utf-8")
            except UnicodeDecodeError:
                print("  Warning: utf-8 failed, trying iso-8859-1...")
                content = response.content.decode("iso-8859-1")

        # Parse CSV
        df = pd.read_csv(StringIO(content), sep=source.delimiter)

        # Cache
        self.data[name] = df
        self._cache_timestamp[name] = datetime.now()

        print(f"  ✓ Loaded {len(df)} rows, {len(df.columns)} columns")

        # Build metadata index if key_column specified
        if source.key_column and source.key_column in df.columns:
            self._build_metadata_index(name, df, source.key_column)

        return df

    def load_all(
        self, force_reload: bool = False, timeout: int = 60
    ) -> dict[str, pd.DataFrame]:
        """
        Load data from all sources

        Args:
            force_reload: Force reload even if cached
            timeout: Request timeout

        Returns:
            dict mapping source name to DataFrame

        Example:
            all_data = loader.load_all()
            for name, df in all_data.items():
                print(f"{name}: {len(df)} rows")
        """
        results = {}
        for name in self.sources:
            try:
                df = self.load_source(name, force_reload, timeout)
                results[name] = df
            except Exception as e:
                print(f"  ✗ Error loading {name}: {e}")

        return results

    def get_all_params(
        self,
        source: str | None = None,
        force_reload: bool = False,
        timeout: int = 60,
    ) -> pd.DataFrame:
        """
        Return a DataFrame with all parameters (all rows/columns).

        Args:
            source:
                - If provided: return the DataFrame for that source.
                - If None:
                    * if only one source exists, return that.
                    * if multiple sources, return a concatenated DataFrame
                      with an extra 'source' column indicating the origin.
            force_reload: if True, reload data from remote even if cached.
            timeout: request timeout passed to load_source/load_all.

        Returns:
            pandas.DataFrame with all parameter rows and columns.
            If no data loaded/available, returns an empty DataFrame.
        """
        # No sources configured
        if not self.sources:
            return pd.DataFrame()

        # If a specific source is requested
        if source is not None:
            # ensure it is loaded
            df = self.load_source(source, force_reload=force_reload, timeout=timeout)
            return df.copy()

        # No specific source: load all
        all_data = self.load_all(force_reload=force_reload, timeout=timeout)

        if not all_data:
            return pd.DataFrame()

        # Single source → just return that
        if len(all_data) == 1:
            return next(iter(all_data.values())).copy()

        # Multiple sources → concatenate and add 'source' column
        frames = []
        for name, df in all_data.items():
            if df is None or df.empty:
                continue
            tmp = df.copy()
            tmp["source"] = name
            frames.append(tmp)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True, sort=False)

    def _build_metadata_index(
        self, source_name: str, df: pd.DataFrame, key_column: str
    ) -> None:
        """Build metadata index from DataFrame"""
        for _, row in df.iterrows():
            key = row[key_column]
            param = ParameterMetadata(source=source_name, data=row.to_dict())
            # Use source prefix to avoid key conflicts
            self.metadata[f"{source_name}:{key}"] = param
            # Also store without prefix if not conflicting
            if key not in self.metadata:
                self.metadata[key] = param

    def get(self, key: str, source: str | None = None) -> ParameterMetadata | None:
        """
        Get parameter by key

        Args:
            key: Parameter key
            source: Source name (optional, searches all if not specified)

        Returns:
            ParameterMetadata or None

        Example:
            param = loader.get('tre200h0')
            if param:
                print(param['parameter_description_en'])
        """
        if source:
            return self.metadata.get(f"{source}:{key}")
        return self.metadata.get(key)

    def get_dataframe(self, source: str) -> pd.DataFrame:
        """Get DataFrame for source"""
        return self.data.get(source)

    def get_all(self, source: str | None = None) -> dict[str, ParameterMetadata]:
        """
        Get all metadata

        Args:
            source: Filter by source (optional)

        Returns:
            dict of all metadata
        """
        if source:
            return {k: v for k, v in self.metadata.items() if v.source == source}
        return self.metadata.copy()

    def search(
        self,
        keyword: str,
        columns: list[str] | None = None,
        source: str | None = None,
        case_sensitive: bool = False,
    ) -> list[ParameterMetadata]:
        """
        Search metadata by keyword

        Args:
            keyword: Search term
            columns: Columns to search in (searches all if None)
            source: Filter by source
            case_sensitive: Case-sensitive search

        Returns:
            list of matching metadata

        Example:
            results = loader.search('temperature')
            for param in results:
                print(param.get('parameter_shortname'))
        """
        results = []
        keyword_search = keyword if case_sensitive else keyword.lower()

        for _, param in self.metadata.items():
            # Filter by source
            if source and param.source != source:
                continue

            # Search in data
            for col, value in param.data.items():
                # Filter by columns
                if columns and col not in columns:
                    continue

                # Convert to string and search
                value_str = str(value) if case_sensitive else str(value).lower()

                if keyword_search in value_str:
                    results.append(param)
                    break

        return results

    def filter(
        self, conditions: dict[str, Any], source: str | None = None
    ) -> list[ParameterMetadata]:
        """
        Filter metadata by conditions

        Args:
            conditions: dict of column: value conditions
            source: Filter by source

        Returns:
            list of matching metadata

        Example:
            # Find all hourly parameters
            hourly = loader.filter({'parameter_granularity': 'H'})
        """
        results = []

        for _, param in self.metadata.items():
            # Filter by source
            if source and param.source != source:
                continue

            # Check all conditions
            match = True
            for col, expected in conditions.items():
                actual = param.get(col)
                if actual != expected:
                    match = False
                    break

            if match:
                results.append(param)

        return results

    def get_column_names(self, source: str) -> list[str]:
        """Get column names for source"""
        df = self.data.get(source)
        return list(df.columns) if df is not None else []

    def get_unique_values(self, source: str, column: str) -> list[Any]:
        """Get unique values for a column"""
        df = self.data.get(source)
        if df is not None and column in df.columns:
            return df[column].unique().tolist()
        return []

    def get_stats(self, source: str) -> dict[str, Any]:
        """Get statistics for source"""
        df = self.data.get(source)
        if df is None:
            return {}

        return {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "memory_usage": df.memory_usage(deep=True).sum(),
            "loaded_at": self._cache_timestamp.get(source),
        }

    def clear_cache(self, source: str | None = None) -> None:
        """Clear cached data"""
        if source:
            if source in self.data:
                del self.data[source]
            if source in self._cache_timestamp:
                del self._cache_timestamp[source]
        else:
            self.data.clear()
            self._cache_timestamp.clear()
            self.metadata.clear()

    def export_to_csv(
        self, source: str, filepath: str, encoding: str = "utf-8", delimiter: str = ","
    ) -> bool:
        """Export source data to CSV"""
        df = self.data.get(source)
        if df is None:
            return False

        df.to_csv(filepath, index=False, encoding=encoding, sep=delimiter)
        return True

    def export_to_json(
        self, source: str, filepath: str, orient: str = "records"
    ) -> bool:
        """Export source data to JSON"""
        df = self.data.get(source)
        if df is None:
            return False

        df.to_json(filepath, orient=orient, indent=2)  # type: ignore[call-overload]
        return True

    def export_metadata_to_dict(self) -> dict[str, dict]:
        """Export all metadata to dictionary"""
        return {key: param.data for key, param in self.metadata.items()}

    def summary(self) -> str:
        """Get loader summary"""
        lines = [
            "MetaParametersLoader Summary",
            "=" * 70,
            f"Sources: {len(self.sources)}",
            f"Loaded: {len(self.data)}",
            f"Metadata entries: {len(self.metadata)}",
            "",
        ]

        for name, source in self.sources.items():
            lines.append(f"Source: {name}")
            lines.append(f"  URL: {source.url[:60]}...")
            lines.append(f"  Encoding: {source.encoding}")
            lines.append(f"  Delimiter: '{source.delimiter}'")

            if name in self.data:
                df = self.data[name]
                lines.append(f"  Loaded: {len(df)} rows, {len(df.columns)} columns")
                if self._cache_timestamp.get(name):
                    lines.append(
                        f"  Cache: {self._cache_timestamp[name].strftime('%Y-%m-%d %H:%M:%S')}"
                    )
            else:
                lines.append("  Status: Not loaded")

            lines.append("")

        return "\n".join(lines)

    def __repr__(self):
        return f"MetaParametersLoader(sources={len(self.sources)}, loaded={len(self.data)})"

    def __str__(self):
        return self.summary()


if __name__ == "__main__":
    print("=" * 70)
    print("MetaParametersLoader - Usage Examples")
    print("=" * 70)

    # Example 1: Initialize and add sources
    print("\n[Example 1] Initialize and Add Sources")
    print("-" * 70)

    loader = MetaParametersLoader()

    # Add MeteoSwiss parameter metadata
    loader.add_source(
        name="Forecast meta parameters",
        url="https://data.geo.admin.ch/ch.meteoschweiz.ogd-local-forecasting/ogd-local-forecasting_meta_parameters.csv",
        description="MeteoSwiss forecast parameters",
        encoding="latin-1",
        delimiter=";",
        key_column="parameter_shortname",
    )

    # Add station metadata
    loader.add_source(
        name="Forecast station id",
        url="https://data.geo.admin.ch/ch.meteoschweiz.ogd-local-forecasting/ogd-local-forcasting_meta_point.csv",
        description="MeteoSwiss stations",
        encoding="latin-1",
        delimiter=";",
        key_column="point_id",
    )

    loader.add_source(
        name="Historic parameters",
        url="https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/ogd-smn_meta_parameters.csv",
        description="MeteoSwiss Historic parameters",
        encoding="latin-1",
        delimiter=";",
        key_column="point_id",
    )

    print(f"✓ Added {len(loader.list_sources())} sources:")
    for name in loader.list_sources():
        print(f"  • {name}")

    # Example 2: Load data
    print("\n[Example 2] Load Data")
    print("-" * 70)

    loader.load_all()

    print("\n[Example 2.1] Load params by sources")
    print("-" * 70)
    df_params_only = loader.get_all_params(source="Historic parameters")

    print(
        f"✓ Loaded parameters DataFrame with {len(df_params_only)} rows, {len(df_params_only.columns)} columns"
    )
    print(f"  Columns: {df_params_only.columns.tolist()}")
    print("\n", df_params_only.head(5))

    # Example 3: Get parameter
    print("\n[Example 3] Get Parameter by Key")
    print("-" * 70)

    param = loader.get("tre200h0")
    if param:
        print(f"✓ Found parameter: {param.get('parameter_shortname')}")
        print(f"  Description: {param.get('parameter_description_en')}")
        print(f"  Unit: {param.get('parameter_unit')}")
        print(f"  Granularity: {param.get('parameter_granularity')}")

    # Example 4: Search
    print("\n[Example 4] Search Parameters")
    print("-" * 70)

    results = loader.search("temperature", source="parameters")
    print(f"✓ Found {len(results)} temperature parameters:")
    for param in results[:5]:
        print(
            f"  • {param.get('parameter_shortname'):12s} - {param.get('parameter_description_en')}"
        )

    # Example 5: Filter
    print("\n[Example 5] Filter by Conditions")
    print("-" * 70)

    hourly = loader.filter({"parameter_granularity": "H"}, source="parameters")
    print(f"✓ Found {len(hourly)} hourly parameters")

    # Example 6: Summary
    print("\n[Example 6] Loader Summary")
    print("-" * 70)
    print(loader.summary())

    print("\n" + "=" * 70)
    print("✓ All examples completed!")
    print("=" * 70)

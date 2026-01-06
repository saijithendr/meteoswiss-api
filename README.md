# meteoswiss-api
FastAPI Application to fetch the historic, forecast weather data in Switzerland.

```
meteoswiss-api/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application factory
│   ├── config.py            # Configuration management
│   ├── dependencies.py      # Dependency injection
│   ├── models/
│   │   ├── __init__.py
│   │   ├── stations.py      # Pydantic models for stations
│   │   ├── parameters.py    # Pydantic models for parameters
│   │   ├── weather.py       # Pydantic models for weather data
│   │   └── responses.py     # Response models
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── stations.py      # /api/v1/stations/ routes
│   │   ├── parameters.py    # /api/v1/parameters/ routes
│   │   ├── weather.py       # /api/v1/weather/ routes
│   │   └── health.py        # /api/v1/health/ routes
│   ├── services/
│   │   ├── __init__.py
│   │   ├── stations_service.py
│   │   ├── parameters_service.py
│   │   └── weather_service.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── cache.py         # Caching utilities
│   │   ├── validators.py    # Input validation
│   │   └── converters.py    # Data conversion (JSON/CSV)
│   └── middleware/
│       ├── __init__.py
│       ├── logging.py       # Request logging
│       ├── error_handler.py # Error handling
│       └── rate_limit.py    # Rate limiting
├── tests/
│   ├── __init__.py
│   ├── test_stations.py
│   ├── test_parameters.py
│   ├── test_weather.py
│   └── conftest.py
├── requirements.txt
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── docs/
│   ├── API.md               # API documentation
│   ├── DEPLOYMENT.md        # Deployment guide
│   └── EXAMPLES.md          # Usage examples
└── README.md
```
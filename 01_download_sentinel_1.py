from cdsetool.query import query_features
from cdsetool.download import download_features
from cdsetool.credentials import Credentials

credentials = Credentials("", "")

features = query_features(
    "Sentinel1",
    {
        "startDate": "2025-04-01",
        "completionDate": "2025-10-31",
        "processingLevel": "LEVEL1",
        "productType": "GRD",
        "sensorMode": "IW",
        "orbitDirection": "ASCENDING",  
        "geometry": "POLYGON((17.9 53.0, 19.1 53.0, 19.1 54.4, 17.9 54.4, 17.9 53.0))",
    }
)

download_features(features, "/data/raw/ascending", credentials=credentials)
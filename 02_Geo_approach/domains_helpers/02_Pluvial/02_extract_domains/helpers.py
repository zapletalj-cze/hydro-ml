"""
This script defines parameters necessary to create basic inputs for the pluvial model.
The script uses functions from other scripts within the module.


Author: Jakub Zapletal
Version: 1.0
Date: 25/03/2026
Date modified: 25/03/2026
"""

import yaml

"""
MATERIALS INPUTS
    P01 - OSM extraction
"""


def func_print(func_name, start=True):
    message = f"▶ {func_name} has started!" if start else f"✓ {func_name}!"
    print(message)


def project_id_lookup(project_name):
    """Returns project_id for a given project name, or None if not found."""
    mapping = {
        "N": 1,
        "D": 2,
        "B": 3,
        "I": 4,
        "F": 5,
        "G": 6,
        "S": 7,
        "Global Flood": 8,
        "Other": 9,
    }
    return mapping.get(project_name, None)


class Parameters:
    def load_local_parameters(path_yaml: str, param_name: str):
        """
        Loads and returns parameters from the specified YAML file
        """
        with open(path_yaml, "r") as file:
            data = yaml.safe_load(file)
            if not isinstance(data, dict):
                raise ValueError("YAML file should contain a dictionary")
            if param_name in data:
                return data[param_name]
            else:
                raise ValueError(f"Parameter '{param_name}' not found in the YAML file")

    def get_local_parameter(params, key):
        try:
            return params.get(key, None)
        except Exception:
            return None

    def set_yaml_file(self, path_yaml):
        self.yaml_path = path_yaml


class OsmExtractSettings:
    outfile_input = {
        "ArableLand_40_OSM.gpkg": "gis_osm_landuse_a_free_1.shp",
        "Forest_10_OSM.gpkg": "gis_osm_landuse_a_free_1.shp",
        "Allotments_63_OSM.gpkg": "gis_osm_landuse_a_free_1.shp",
        "Meadow_62_OSM.gpkg": "gis_osm_landuse_a_free_1.shp",
        "Grass_61_OSM.gpkg": "gis_osm_landuse_a_free_1.shp",
        "Sand_86_OSM.gpkg": "gis_osm_landuse_a_free_1.shp",
        "Shrub_20_OSM.gpkg": "gis_osm_landuse_a_free_1.shp",
        "Wetland_90_OSM.gpkg": "gis_osm_water_a_free_1.shp",
        "RockyTerrain_60_OSM.gpkg": "gis_osm_landuse_a_free_1.shp",
        "Parking_58_OSM.gpkg": "gis_osm_traffic_a_free_1.shp",
        "Railway_58_OSM.gpkg": "gis_osm_railways_free_1.shp",
        "RoadsSecondary_56_OSM.gpkg": "gis_osm_roads_free_1.shp",
        "RoadsPrimary_55_OSM_buffer7_5.gpkg": "gis_osm_roads_free_1.shp",
        "RoadsPrimary_55_OSM_buffer3.gpkg": "gis_osm_roads_free_1.shp",
        "WaterBodies_80_OSM.gpkg": "gis_osm_water_a_free_1.shp",
        "Waterways_80_OSM.gpkg": "gis_osm_waterways_free_1.shp",
        "Buildings_51_OSM.gpkg": "gis_osm_buildings_a_free_1.shp",
        "Airports_59_OSM.gpkg": "gis_osm_transport_a_free_1.shp",
    }
    outfile_filtration = {
        "Railway_58_OSM.gpkg": ["bridge", "tunnel"],
        "RoadsSecondary_56_OSM.gpkg": ["bridge", "tunnel"],
        "RoadsPrimary_55_OSM_buffer7_5.gpkg": ["bridge", "tunnel"],
        "RoadsPrimary_55_OSM_buffer3.gpkg": ["bridge", "tunnel"],
    }
    # 1:n
    outfile_classes = {
        "ArableLand_40_OSM.gpkg": ["farmland"],
        "Forest_10_OSM.gpkg": ["forest", "park", "orchard"],
        "Allotments_63_OSM.gpkg": ["allotments"],
        "Buildings_51_OSM.gpkg": ["building"],
        "Meadow_62_OSM.gpkg": ["meadow"],
        "Grass_61_OSM.gpkg": ["grass"],
        "Sand_86_OSM.gpkg": ["beach", "sand"],
        "Shrub_20_OSM.gpkg": ["scrub", "scree", "heath"],
        "Wetland_90_OSM.gpkg": ["wetland"],
        "RockyTerrain_60_OSM.gpkg": ["rock", "stone", "cliff"],
        "Parking_58_OSM.gpkg": ["parking"],
        "Railway_58_OSM.gpkg": ["rail", "light_rail"],
        "RoadsSecondary_56_OSM.gpkg": ["service", "road", "unclassified"],
        "RoadsPrimary_55_OSM_buffer7_5.gpkg": [
            "motorway",
            "trunk",
            "raceway",
            "trunk_link",
            "motorway_link",
        ],
        "RoadsPrimary_55_OSM_buffer3.gpkg": [
            "trunk_link",
            "primary",
            "primary_link",
            "secondary",
            "secondary_link",
            "tertiary",
            "tertiary_link",
            "residential",
            "living_street",
        ],
        "WaterBodies_80_OSM.gpkg": ["water", "reservoir", "riverbank"],
        "Waterways_80_OSM.gpkg": ["river", "stream", "ditch", "canal"],
        "Airports_59_OSM.gpkg": ["airport", "apron"],
    }

    # 1:1
    # defines basic buffer of lines - hardcoded below
    outfile_postprocess = {
        "RoadsPrimary_55_OSM_buffer7_5.gpkg": "buffer7_5",
        "RoadsPrimary_55_OSM_buffer3.gpkg": "buffer3",
    }

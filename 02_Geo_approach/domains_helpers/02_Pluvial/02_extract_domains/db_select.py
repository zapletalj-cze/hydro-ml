"""
Helper class to insert dataset rows into the EUFL catalogue database.
The script uses functions from other scripts within the module.


Author: Jakub Zapletal
Version: 1.0
Date: 22/03/2026
Date modified: 22/03/2026
"""

import getpass
import warnings

import sqlalchemy as sal

warnings.filterwarnings("ignore")

DB_SERVER = "eupraappp104"
DB_PORT = "5432"
DB_DATABASE = "eufl_catalogue"
DB_USER = getpass.getuser().lower()


def _build_engine(password):
    conn_url = sal.engine.URL.create(
        drivername="postgresql+psycopg2",
        username=DB_USER,
        password=password,
        host=DB_SERVER,
        port=int(DB_PORT),
        database=DB_DATABASE,
        query={"sslmode": "disable"},
    )
    return sal.create_engine(conn_url)


def verify_password(password: str) -> bool:
    """Return True if the password successfully authenticates against the DB."""
    try:
        with _build_engine(password).connect():
            pass
        return True
    except Exception as e:
        print(f"[verify_password] Database connection failed: {e}")
        return False


class CatalogueSelect:
    """Select dataset rows from eufl_catalogue.dataset on the EUFL catalogue PostgreSQL/PostGIS database."""

    def __init__(self, type_id, password):
        self.type_id = type_id
        self.engine = _build_engine(password)

    def _get_dtm_footprints(self, wkt_polygon, cell_size=0):
        if cell_size == "high_res":
            sql = """
                SELECT link FROM eufl_catalogue.dataset
                WHERE type_id = 1 AND ST_Intersects(geometry, ST_GeomFromText(%s, 3035))
                AND resolution < 5
            """
            params = (wkt_polygon,)
        else:
            sql = """
                SELECT link FROM eufl_catalogue.dataset
                WHERE type_id = 1 AND ST_Intersects(geometry, ST_GeomFromText(%s, 3035))
                AND resolution = %s
            """
            params = (wkt_polygon, cell_size)
        with self.engine.connect() as conn:
            result = conn.exec_driver_sql(sql, params).fetchall()
        return [row[0] for row in result]

    def _link_exists(self, link: str) -> bool:
        """Return True if a row with this exact link already exists in eufl_catalogue.dataset."""
        sql = "SELECT 1 FROM eufl_catalogue.dataset WHERE link = %s LIMIT 1"
        with self.engine.connect() as conn:
            result = conn.exec_driver_sql(sql, (link,)).scalar()
        return result is not None

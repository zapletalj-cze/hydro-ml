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


class CatalogueWriter:
    """Insert a single row into eufl_catalogue.dataset on the EUFL catalogue PostgreSQL/PostGIS database."""

    _SQL_WITH_GEOM = """
        INSERT INTO eufl_catalogue.dataset
            (name, type_id, project_id, area_id, link, geometry,
             epsg_code, date_created, added_by, data_source,
             resolution, version, notes)
        VALUES
            (%s, %s, %s, %s, %s, ST_GeomFromText(%s, 3035),
             %s, %s, %s, %s, %s, %s, %s)
    """
    _SQL_NO_GEOM = """
        INSERT INTO eufl_catalogue.dataset
            (name, type_id, project_id, area_id, link, geometry,
             epsg_code, date_created, added_by, data_source,
             resolution, version, notes)
        VALUES
            (%s, %s, %s, %s, %s, NULL,
             %s, %s, %s, %s, %s, %s, %s)
    """

    def __init__(self, type_id, password, epsg_code=3035):
        self.type_id = int(type_id)
        self.epsg_code = epsg_code
        self.added_by = getpass.getuser()
        self.engine = _build_engine(password)

    def _get_next_max_version(self, area_id, type_id):
        """Get max version from database for one dataset type and area, returns 0 if no existing dataset found."""
        sql = """
            SELECT MAX(version) FROM eufl_catalogue.dataset
            WHERE type_id = %s AND area_id = %s
        """
        with self.engine.connect() as conn:
            result = conn.exec_driver_sql(sql, (type_id, area_id)).scalar()
        return result + 1 if result is not None else 0

    def _get_version(self, dataset_id=None, link=None):
        """Get version from database for a specific dataset ID or link"""
        if dataset_id:
            sql = """
                SELECT version FROM eufl_catalogue.dataset
                WHERE dataset_id = %s
            """
            params = (dataset_id,)
        elif link:
            sql = """
                SELECT version FROM eufl_catalogue.dataset
                WHERE link = %s
            """
            params = (link,)
        else:
            raise ValueError("Either 'dataset_id' or 'link' must be provided.")
        with self.engine.connect() as conn:
            result = conn.exec_driver_sql(sql, params).scalar()
        return result

    def write(
        self,
        name,
        project_id,
        area_id,
        link,
        resolution,
        version,
        notes,
        geom_wkt=None,
        date_created=None,
    ):
        """Insert one row. Raises ValueError if name is missing."""
        if not name:
            raise ValueError("'name' is required.")

        notes = notes or ""
        if geom_wkt:
            sql = self._SQL_WITH_GEOM
            params = (
                str(name),
                self.type_id,
                project_id,
                area_id,
                link,
                geom_wkt,
                self.epsg_code,
                date_created,
                self.added_by,
                None,
                resolution,
                version,
                notes,
            )
        else:
            sql = self._SQL_NO_GEOM
            params = (
                str(name),
                self.type_id,
                project_id,
                area_id,
                link,
                self.epsg_code,
                date_created,
                self.added_by,
                None,
                resolution,
                version,
                notes,
            )
        with self.engine.begin() as conn:
            conn.exec_driver_sql(sql, params)
        print(f"✓ Inserted: {name}")

import psycopg2
from psycopg2 import sql

def get_connection(
    dbname,
    user,
    password,
    host='localhost',
    port=5432
):
    """
    Establishes and returns a connection to the PostgreSQL database.
    """
    conn = psycopg2.connect(
        dbname=dbname,
        user=user,
        password=password,
        host=host,
        port=port
    )
    return conn

def create_function_calls_table(conn, table_name):
    """
    Creates a sample table if it does not exist.
    """
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("""
                CREATE TABLE FUNCTION CALLS IF NOT EXISTS {} (
                    id SERIAL PRIMARY KEY,
                    user_input TEXT NOT NULL
                    function_name VARCHAR(255) NOT NULL,
                    function_args TEXT NOT NULL;
                    sucess BOOLEAN NOT NULL;
                    result TEXT;
                    used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

                )
            """).format(sql.Identifier(table_name))
        )
        conn.commit()
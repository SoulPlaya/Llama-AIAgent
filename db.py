import psycopg2
from psycopg2 import sql



class Database: 

    def __init__(self):
        self.get_connection()
        self.create_functions_table()
        self.create_function_arguments_table()
        self.create_function_calls_table()
    

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

    def create_functions_table(conn):
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("""
                    CREATE TABLE FUNCTIONS IF NOT EXISTS {} (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    );""")
            )
            conn.commit()
        
    def create_function_arguments_table(conn):
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("""
                    CREATE TABLE FUNCTION ARGUMENTS IF NOT EXISTS {} (
                    id SERIAL PRIMARY KEY,
                    function_id INTERGER REFERENCES functions(id) ON DELETE CASCADE,
                    arg_type TEXT NOT NULL,
                    is_required BOOLEAN,
                    defualt_value TEXT,
                    description TEXT,
                    arg_order INTEGER,
                    UNIQUE(function_id, arg_name)
                    );
                    """)
            )
            conn.commit()

    def get_function_names(conn):
        query = 'SELECT name FROM Functions'
        with conn.cursor() as cur:
            cur.execute(query)

    def function_name_to_id(self, conn, name):
        query = f'SELECT * from functions where name = {name}'
        with conn.cursor() as cur:
            cur.execute(query)
        

    def get_function_args_by_name(self, conn, name):
        id = self.function_name_to_id(name)
        query = f'SELECT * FROM '
    

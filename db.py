import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv
import os

load_dotenv()
host = os.getenv('POSTGRES_HOST')
port = os.getenv('POSTGRES_PORT')
user = os.getenv('POSTGRES_USER')
password = os.getenv('POSTGRES_PASSWORD')
dbname = os.getenv('POSTGRES_DB')

class Database:
    def __init__(self):
        self.conn = self.get_connection(dbname, user, password, host, port)
        self.create_functions_table()
        self.create_function_arguments_table()
        self.create_function_calls_table()
   
    def get_connection(self, dbname, user, password, host='localhost', port=5432):
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
    
    def create_function_calls_table(self):
        """
        Creates function_calls table if it does not exist.
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS function_calls (
                    id SERIAL PRIMARY KEY,
                    user_input TEXT NOT NULL,
                    function_name VARCHAR(255) NOT NULL,
                    function_args TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    result TEXT,
                    used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.conn.commit()
    
    def create_functions_table(self):
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS functions (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.conn.commit()
       
    def create_function_arguments_table(self):
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS function_arguments (
                    id SERIAL PRIMARY KEY,
                    function_id INTEGER REFERENCES functions(id) ON DELETE CASCADE,
                    arg_name TEXT NOT NULL,
                    arg_type TEXT NOT NULL,
                    is_required BOOLEAN,
                    default_value TEXT,
                    description TEXT,
                    arg_order INTEGER,
                    UNIQUE(function_id, arg_name)
                )
            """)
            self.conn.commit()

    def seed_database(self):
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM functions")
            count = cur.fetchone()[0]
            if count == 0:
                cur.execute("""
                    INSERT INTO functions (name, description) VALUES
                    ('scan_ports', 'scans selected ports on a selected IP address'),
                    ('take_screenshot', 'takes a screenshot of the screen for processes like ai-decription of on screen events'),
                    ('search_web', 'searches the web for a given query')
                """)
                self.conn.commit()
            cur.execute("SELECT COUNT(*) FROM function_arguments")
            count = cur.fetchone()[0]
            if count == 0:
                cur.execute("""
                    INSERT INTO function_arguments (function_id, arg_name, arg_type, is_required, default_value, description, arg_order) VALUES
                            
                    ((SELECT id FROM functions WHERE name='scan_ports'), 'target', 'str', TRUE, NULL, 'The target IP address or hostname to scan', 1),
                    ((SELECT id FROM functions WHERE name='scan_ports'), 'ports', 'str', FALSE, '1-1024', 'The range of ports to scan (default: 1-1024)', 2),
                            
                    ((SELECT id FROM functions WHERE name='search_web'), 'query', 'str', TRUE, NULL, 'The search query string', 1)
                """)
                self.conn.commit()
            
    
    def get_function_names(self):
        query = 'SELECT name FROM functions'
        with self.conn.cursor() as cur:
            cur.execute(query)
            return cur.fetchall()
    
    def get_function_name_to_id(self, name):
        query = 'SELECT id FROM functions WHERE name = %s'
        with self.conn.cursor() as cur:
            cur.execute(query, (name,))
            result = cur.fetchone()
            return result[0] if result else None
       
    def get_function_args_by_name(self, name):
        function_id = self.get_function_name_to_id(name)
        if function_id is None:
            return None
        query = 'SELECT * FROM function_arguments WHERE function_id = %s ORDER BY arg_order'
        with self.conn.cursor() as cur:
            cur.execute(query, (function_id,))
            return cur.fetchall()
        
    def get_function_calls(self):
        query = 'SELECT * FROM function_calls ORDER BY used_at DESC'
        with self.conn.cursor() as cur:
            cur.execute(query)
            return cur.fetchall()
        
    
    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()
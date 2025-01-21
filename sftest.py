from snowflake_connection import conn
import snowflake.connector

def get_connection():
    # Define the connection parameters
    conn_params = {
        'user': 'your_username',
        'password': 'your_password',
        'account': 'your_account',
        'warehouse': 'your_warehouse',
        'database': 'your_database',
        'schema': 'your_schema'
    }

    # Establish the connection
    conn = snowflake.connector.connect(**conn_params)
    return conn

def execute_query(query):
    conn = get_connection()
    cur = conn.cursor()
    
    try:
        # Execute the query
        cur.execute(query)
        
        # Fetch the results
        results = cur.fetchall()
        
        return results
    finally:
        # Close the cursor and connection
        cur.close()
        conn.close()

if __name__ == "__main__":
    query = "SELECT * FROM your_table LIMIT 10"
    results = execute_query(query)
    
    # Print the results
    for row in results:
        print(row)

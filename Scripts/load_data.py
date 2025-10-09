import os
import snowflake.connector
from dotenv import load_dotenv

load_dotenv()

conn=snowflake.connector.connect(
    user=os.getenv('SNOWFLAKE_USER'),
    password=os.getenv('SNOWFLAKE_PASSWORD'),
    account=os.getenv('SNOWFLAKE_ACCOUNT'),
    warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
    database=os.getenv('SNOWFLAKE_DATABASE'),
    schema=os.getenv('SNOWFLAKE_SCHEMA')
)

cur=conn.cursor()

queries = {
    "Version": "SELECT CURRENT_VERSION()",
    "User": "SELECT CURRENT_USER()",
    "Account": "SELECT CURRENT_ACCOUNT()"
}

# Execute and print results
for key, query in queries.items():
    cur.execute(query)
    result = cur.fetchone()[0]
    print(f"{key}: {result}")

cur.close()
conn.close()

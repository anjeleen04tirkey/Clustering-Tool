import snowflake as sf
import snowflake.connector
from snowflake.sqlalchemy import URL
from sqlalchemy import create_engine, exc
import os
import pandas as pd
import numpy as np
from datetime import datetime,date
import logging
 
 
SF_DB_ACCOUNT = 'kmartau.ap-southeast-2'
SF_DB_USER = 'anjeleen.tirkey@anko.com'
SF_DB_ROLE = 'KSF_DATAANALYTICS'
SF_DB_WAREHOUSE = 'KSF_DATA_SCIENTIST_WH'
SF_DB_DATABASE = 'KSF_SOPHIA_DATA_INTELLIGENCE_HUB_PROD'
 
def get_settings():
        """ Function to get database settings """
        settings = {
            "account": SF_DB_ACCOUNT,
            "user": SF_DB_USER,
            "database": SF_DB_DATABASE,
            "warehouse": SF_DB_WAREHOUSE,
            "role": SF_DB_ROLE
        }
        return settings
 
def generate_connection():
    """ Function to generate database connection """
    settings = get_settings()
    engine = create_engine(URL(
        account=settings['account'],
        user=settings['user'],
        authenticator='externalbrowser',
        database=settings['database'],
        warehouse=settings['warehouse'],
        role=settings['role']))
    return engine
 
engine = generate_connection()

query = '''SELECT TOP 10 * FROM ksf_sophia_data_intelligence_hub_prod.sales.fact_sales_detail'''

with engine.connect() as connection:
    data = pd.read_sql(query,connection)

print(data)
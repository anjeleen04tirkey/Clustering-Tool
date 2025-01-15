import os
from sqlalchemy import create_engine
#from sqlalchemy import URL
from sqlalchemy.engine import URL

SF_DB_ACCOUNT = 'kmartau.ap-southeast-2.privatelink'

SF_DB_USER = 'anjeleen.tirkey@anko.com'

SF_DB_ROLE = 'KSF_DATAANALYTICS'

SF_DB_WAREHOUSE = 'KSF_DATA_SCIENTIST_WH (X-SMALL)'

SF_DB_DATABASE = 'KSF_SOPHIA_DATA_INTELLIGENCE_HUB_PROD'

SF_DB_SCHEMA = 'SALES'



SF_DB_AUTHENTICATOR = 'externalbrowser'




import pandas as pd
from snowflake.sqlalchemy import URL
from sqlalchemy import create_engine
import os

def get_conn():
    engine = create_engine(URL(
        account='kmartau.ap-southeast-2.privatelink',
        database="KSF_SOPHIA_DATA_INTELLIGENCE_HUB_PROD",
        warehouse="KSF_DATA_SCIENTIST_WH (X-SMALL)",
        role='KSF_DATAANALYTICS',
        authenticator='externalbrowser',
        user='anjeleen.tirkey@anko.com'
    ))
    print(f'connection done, engine returned')
    return engine

engine = get_conn()



data = pd.read_sql("select top 10 * from ksf_sophia_data_intelligence_hub_prod.sales.fact_sales_detail",engine)
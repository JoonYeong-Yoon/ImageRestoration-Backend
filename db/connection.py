import pymysql
from config.settings import DB_CONFIG

def get_connection():
    return pymysql.connect(
        **DB_CONFIG,
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True
    )

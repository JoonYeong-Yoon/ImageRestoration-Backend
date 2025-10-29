import pymysql
from config.settings import DB_CONFIG

def get_connection():
    """MySQL DB 커넥션 생성"""
    return pymysql.connect(
        **DB_CONFIG,
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True
    )


def test_connection():
    """DB 연결 테스트용 함수"""
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1 AS result")
            result = cursor.fetchone()
            print("✅ DB 연결 성공:", result)
        conn.close()
    except Exception as e:
        print("❌ DB 연결 실패:", e)


def execute_query(sql, params=None, fetchone=False, fetchall=False):
    """
    공통 SQL 실행 함수
    - INSERT / UPDATE / DELETE / SELECT 등 다 가능
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute(sql, params)
            if fetchone:
                return cursor.fetchone()
            elif fetchall:
                return cursor.fetchall()
            else:
                return True
    except Exception as e:
        print("❌ SQL 실행 에러:", e)
        return None
    finally:
        if conn:
            conn.close()

# 챗봇 데이터 학습용 테이블을 생성하는 코드
## 해당 소스는 챗봇 학습툴과 관련있기 때문에 /train_tools/qna 디렉터리 내에 생성해야 함

import pymysql
from config.DatabaseConfig import *    # DB 접속 정보 불러오기

db = None
try:
    db = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        passwd=DB_PASSWORD,
        db=DB_NAME,
        charset='utf8'
    )

    # 테이블 생성 sql 정의
    sql = '''
      CREATE TABLE IF NOT EXISTS `chatbot_train_data` (
      `id` INT UNSIGNED NOT NULL AUTO_INCREMENT,
      `intent` VARCHAR(45) NULL,
      `ner` VARCHAR(1024) NULL,
      `query` TEXT NULL,
      `answer` TEXT NOT NULL,
      `answer_image` VARCHAR(2048) NULL,
      PRIMARY KEY (`id`))
    ENGINE = InnoDB DEFAULT CHARSET=utf8
    '''

    # 테이블 생성
    with db.cursor() as cursor:
        cursor.execute(sql)

except Exception as e:
    print(e)

finally:
    if db is not None:
        db.close()
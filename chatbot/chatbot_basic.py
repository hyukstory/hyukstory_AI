# < 데이터 베이스 연결하기 >
import pymysql

db = None
try:
    #DB 호스트 정보에 맞게 입력
    db = pymysql.connect(           # pymysql.connect() 함수를 사용하면 DB 서버에 접속할 수 있다.
        host='127.0.0.1',           # 데이터 베이스 서버가 존재하는 호스트 주소
        user='root',                # 데이터베이스 로그인 유저
        passwd='hyukstory',          # 데이터베이스 로그인 패스워드
        db='chatbot',             # 데이터베이스 명
        charset='utf8'              # 데이터베이스에서 사용할 charset 인코딩
    )
    print("DB 연결 성공 ")

except Exception as e:
    print(e) #db 연결 실패 시 오류 내용 출력

finally:
    if db is not None: #db가 연결된 경우에만 접속 닫기 시도
        db.close()
        print("DB 연결 닫기 성공")
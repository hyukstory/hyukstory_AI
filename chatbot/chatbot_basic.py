# < 데이터 베이스 연결하기 >
import pymysql

db = None
try:
    #DB 호스트 정보에 맞게 입력
    db = pymysql.connect(           # pymysql.connect() 함수를 사용하면 DB 서버에 접속할 수 있다.
        host='127.0.0.1',           # 데이터 베이스 서버가 존재하는 호스트 주소
        user='root',                # 데이터베이스 로그인 유저
        passwd='hyukstory',         # 데이터베이스 로그인 패스워드
        db='chatbot',               # 데이터베이스 명
        charset='utf8'              # 데이터베이스에서 사용할 charset 인코딩
    )
    print("DB 연결 성공 ")

except Exception as e:
    print(e) #db 연결 실패 시 오류 내용 출력

finally:
    if db is not None: #db가 연결된 경우에만 접속 닫기 시도
        db.close()
        print("DB 연결 닫기 성공")




# < 데이터 조작하기 1. 데이터테이블 만들기 >
import pymysql
db = None
try:
    #DB 호스트 정보에 맞게 입력
    db = pymysql.connect(           # pymysql.connect() 함수를 사용하면 DB 서버에 접속할 수 있다.
        host='127.0.0.1',           # 데이터 베이스 서버가 존재하는 호스트 주소
        user='root',                # 데이터베이스 로그인 유저
        passwd='hyukstory',         # 데이터베이스 로그인 패스워드
        db='chatbot',               # 데이터베이스 명
        charset='utf8'              # 데이터베이스에서 사용할 charset 인코딩
    )
    print("DB 연결 성공 ")

    # 테이블 삽입 sql 정의 --- 1)
    sql = '''
        CREATE TABLE tb_student(                     
            id int primary key auto_increment not null, 
            name varchar(32),
            age int,
            address varchar(32)
            ) ENGINE = InnoDB DEFAULT CHARSET=utf8
            '''

        # tb_student 라는 테이블 생성
        # 컬럼 명 id는 기본키, null 일수 없는 제약 조건을 갖는다
        # 칼럼명 name은 32자 내외의 가변 길이의 문자열을 받는 제약 조건
        # 칼럼명 age는 정수를 받는 제약 조건
        # 칼렁명 address는 32자 내외의 가변 길이의 문자열을 받는 제약 조건
        # DB 테이블을 생성할 때 사용되는 기본 설정

    # 테이블 생성 --- 2)
    with db.cursor() as cursor :        # 연결한 DB와 상호작용 하려면 cursor 객체 필요
        cursor.execute(sql)             # cursor 객체는 우리가 임의로 생성할 수 없으며
                                        # 반드시 DB 호스트에 연결된 객체(db)의 cursor() 함수로 cursor 객체를 받아와야 함
                                        # cursor 객체의 execute() 함수로 SQL 구문을 실행
                                        # with 구문 내에서 cursor 객체를 사용하기 때문에 사용 후에는 자동으로 메모리에서 해제

except Exception as e:
    print(e) #db 연결 실패 시 오류 내용 출력

finally:
    if db is not None: #db가 연결된 경우에만 접속 닫기 시도
        db.close()
        print("DB 연결 닫기 성공")



# < 데이터 조작하기 2. 데이터 삽입 >
import pymysql
db = None
try:
    #DB 호스트 정보에 맞게 입력
    db = pymysql.connect(
        host='127.0.0.1',
        user='root',
        passwd='hyukstory',
        db='chatbot',
        charset='utf8'
    )
    print("DB 연결 성공 ")

    # 데이터 삽입 sql 정의 --- 1)
    sql = '''
    INSERT tb_student(name, age, address) values('hyuksoo', 28, 'Korea') 
        '''
            # 앞서 생성한 tb_student 테이블에 데이터를 삽입하기 위해 정의한 SQL 구문

    # 데이터 삽입 --- 2)
    with db.cursor() as cursor :        # 연결한 DB와 상호작용 하려면 cursor 객체 필요
        cursor.execute(sql)
    db.commit()                         # DB 호스트에 연결된 객체(db)에 commit()를 통해 수정된 내용을 DB 에 반영

except Exception as e:
    print(e)

finally:
    if db is not None:
        db.close()
        print("DB 연결 닫기 성공")





# < 데이터 조작하기 3. 데이터 변경 >
import pymysql
db = None
try:
    #DB 호스트 정보에 맞게 입력
    db = pymysql.connect(
        host='127.0.0.1',
        user='root',
        passwd='hyukstory',
        db='chatbot',
        charset='utf8'
    )
    print("DB 연결 성공 ")

    # 데이터 수정 sql 정의 --- 1)
        # 업데이트를 한다. where는 찾아주는 기능. id가 %d % id 인 값에서 set name, age 수정
    id = 1             # 데이터 id (Primary Key)
    sql = '''
    UPDATE tb_student set name = "LEEHYUKSOO", age = 20 where id = %d 
        ''' %id
        # 업데이트 수행. tb_student 셋팅.name 을 'LEEHYUKSOO‘ 로, age를 20으로 업데이트.
        # id가 d인 값을 찾아(where) 셋팅된 값으로 업데이트.

    # 데이터 수정 --- 2)
    with db.cursor() as cursor :        # 연결한 DB와 상호작용 하려면 cursor 객체 필요
        cursor.execute(sql)
    db.commit()                         # DB 호스트에 연결된 객체(db)에 commit()를 통해 수정된 내용을 DB 에 반영

except Exception as e:
    print(e)

finally:
    if db is not None:
        db.close()
        print("DB 연결 닫기 성공")




# < 데이터 조작하기 4. 데이터 삭제 >
import pymysql
db = None
try:
    #DB 호스트 정보에 맞게 입력
    db = pymysql.connect(
        host='127.0.0.1',
        user='root',
        passwd='hyukstory',
        db='chatbot',
        charset='utf8'
    )
    print("DB 연결 성공 ")

    # 데이터 삭제 sql 정의 --- 1)
    id = 1  # 데이터 id(Primary Key)
    sql = '''
            DELETE from tb_student where id=%d
        ''' % id

    # 데이터 삭제 --- 2)
    with db.cursor() as cursor :        # 연결한 DB와 상호작용 하려면 cursor 객체 필요
        cursor.execute(sql)
    db.commit()                         # DB 호스트에 연결된 객체(db)에 commit()를 통해 수정된 내용을 DB 에 반영

except Exception as e:
    print(e)

finally:
    if db is not None:
        db.close()
        print("DB 연결 닫기 성공")




# < 데이터 조작하기 5. 다수의 데이터 삽입, 조회 >
import pymysql
import pandas as pd
db = None
try:
    #DB 호스트 정보에 맞게 입력
    db = pymysql.connect(
        host='127.0.0.1',
        user='root',
        passwd='hyukstory',
        db='chatbot',
        charset='utf8'
    )
    print("DB 연결 성공 ")

    # 데이터 DB에 추가될 항목들 정리 ---①
    students = [
        {'name': '철수', 'age': 36, 'address': '서울'},
        {'name': '영희', 'age': 22, 'address': '안양'},
        {'name': '민지', 'age': 31, 'address': '용인'},
        {'name': '주연', 'age': 27, 'address': '부산'},
    ]
    for s in students:  # 한 줄씩 읽어와 s에 저장
        with db.cursor() as cursor:
            sql = '''
                    insert tb_student(name, age, address) values("%s", "%d", "%s")
                    ''' % (s['name'], s['age'], s['address'])
            cursor.execute(sql)
    db.commit()  # 변경사항 저장

    # 30대 학생만 조회 ---②
    cond_age = 30
    with db.cursor(pymysql.cursors.DictCursor) as cursor:
        sql = '''
        select * from tb_student where age > %d 
        ''' % cond_age
        cursor.execute(sql)
        age_results = cursor.fetchall()  # select구문으로 조회한 모든 데이터를 불러오는 함수
    print("30대 이상 : ", age_results)  # 가져온 데이터 출력

    # 보기 불편하므로 pandas 데이터 프레임으로 표현
    df = pd.DataFrame(age_results)
    print(df)

    # 이름검색
    cond_name = '철수'  # 찾고자 하는 이름
    with db.cursor(pymysql.cursors.DictCursor) as cursor:
        sql = '''
        select * from tb_student where name="%s"
        ''' % cond_name
        cursor.execute(sql)
        name_result = cursor.fetchone()  # select 구문으로 조회한 데이터 중 하나만 불러오는 함수
    print('이름 검색 : ', name_result['name'], name_result['age'])

    # pandas 데이터 프레임으로 표현
    df = pd.DataFrame(name_result)
    print(df)

except Exception as e:
    print(e)

finally:
    if db is not None:
        db.close()
        print("DB 연결 닫기 성공")
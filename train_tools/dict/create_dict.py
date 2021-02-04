# 챗봇에서 사용하는 사전 파일 생성
from utils.Preprocess import Preprocess
from tensorflow.keras import preprocessing
import pickle

# 말뭉치 데이터 읽어오기
def read_corpus_data(filename):
    with open(filename, 'r', encoding = 'utf8') as f:
        data = [line.split('\t') for line in f.read().splitlines()]  # 한줄씩 읽어온 후 tap으로 구분
        data = data[1:] # 헤더 건너뛰기
    return data


# 말뭉치 데이터 가져오기
corpus_data = read_corpus_data('C:/Users/hyukstory/Desktop/github/hyukstory_AI/train_tools/dict/corpus.txt')

print(corpus_data[:5])



# 말뭉치 데이터에서 키워드만 추출해서 사전 리스트 생성
p = Preprocess() # 클래스 변수화
dict = []
for c in corpus_data:
    pos = p.pos(c[1])   # 형태소 분석기 pos  태거 / 문장 정보가 담긴 [1]을 선택해서 형태소 분석
    print('pos :', pos)
    for k in pos:       # pos 결과를 k에
        dict.append(k[0]) # pos 태그 결과를 dict에 저장



# 사전에 사용될 word2index 생성
# 사전의 첫번 째 인덱스에는 OOV 사용
tokenizer = preprocessing.text.Tokenizer(oov_token='OOV') # 미리 지정되지 않은 단어들을 OOV 로 지정
tokenizer.fit_on_texts(dict) # fit_on_texts 는 입력한 텍스트로부터 단어 빈도수가 높은 순으로 낮은 정수 인덱스를 부여
                            # 정확히 정수 인코딩 작업이 이루어진다고 보면 됨
word_index = tokenizer.word_index

# 사전 파일 생성
f = open("C:/Users/hyukstory/Desktop/github/hyukstory_AI/train_tools/dict/chatbot_dict.bin", "wb")
try:
    pickle.dump(word_index, f)  # 피클이란 텍스트 상태의 데이터가 아닌 파이썬 객체 자체를 파일로 저장
except Exception as e:
    print(e)
finally:
    f.close()
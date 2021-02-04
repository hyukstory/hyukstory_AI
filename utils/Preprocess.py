from konlpy.tag import Komoran
import pickle
import jpype

class Preprocess:
    def __init__(self, word2index_dic='', userdic=None):
        # 단어 인덱스 사전 불러오기
        if(word2index_dic != ''):               # 만약 word2index_dic 인자값이 있다면, 아래 코드 수행
            f = open(word2index_dic, "rb")      # 받아진 값 word2index_dic를 열고 읽는다
            self.word_index = pickle.load(f)    # 피클로 읽어와 word_index 초기화
            f.close()
        else:
            self.word_index = None

        # 형태소 분석기 초기화
        self.komoran = Komoran(userdic=userdic)

        # 제외할 품사
        # 참조 : https://docs.komoran.kr/firststep/postypes.html
        # 관계언 제거, 기호 제거
        # 어미 제거
        # 접미사 제거
        self.exclusion_tags = [
            'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ',
            'JX', 'JC',
            'SF', 'SP', 'SS', 'SE', 'SO',
            'EP', 'EF', 'EC', 'ETN', 'ETM',
            'XSN', 'XSV', 'XSA'
        ]

    # 형태소 분석기 POS 태거
    def pos(self, sentence):
        jpype.attachThreadToJVM()
        return self.komoran.pos(sentence)      # pos() 함수를 정의하는데, sentence 를 받아서 형태소 분석기를 수행
                                            # 주어진 문장에 대하여 전부 형태소 분석을 수행
                                            # 이렇게 하면 다양한 형태소 분석기를 사용할 수 있으므로 유지보수 측면에서 우수


    # 불용어 제거 후, 필요한 품사 정보만 가져오기 / 람다(익명함수)
    def get_keywords(self, pos, without_tag=False):
        f = lambda x: x in self.exclusion_tags  # 익명함수 f() 정의 : 입력 X가 불용어에 있는지 확인
        word_list = []
        # pos 에 저장된 형태소 분석 결과를 하나씩 넘겨준다
        for p in pos:        # pos에 저장된 형태소 분석 결과를 하나씩 p로 넘겨준다
            if f(p[1]) is False:  # 만약 f()결과가 false면(불용어가 포함되어 있지 않다면, 즉 원하는 키워드들로 잘 되어있다면)
                # 넘겨진 단어의 품사정보를 word_list에 저장, 이때 without_tag가 False가 아닌 참이라면 앞의 단어까지 같이 넘겨주기
                word_list.append(p if without_tag is False else p[0])
        return word_list



    # 키워드를 단어 인덱스 시퀀스로 변환
    def get_wordidx_sequence(self, keywords):
        if self.word_index is None:
            return []

        w2i = []
        for word in keywords:
            try:
                w2i.append(self.word_index[word]) #word_index['단어'] 의 인덱스 값을 w2i에 추가
            except KeyError:
                w2i.append(self.word_index['OOV']) # 해당 단어가 사전에 없는 경우, OOV 처리
        return w2i

from utils.Preprocess import Preprocess

sent = "내일 오전 10시에 짬뽕 주문하고 싶어ㅋㅋ"
p = Preprocess(userdic='C:/Users/hyukstory/Desktop/github/hyukstory_AI/utils/user_dic.tsv')

pos = p.pos(sent)
print(pos)


keywords_F = p.get_keywords(pos, without_tag=False)
print(keywords_F)
keywords_T = p.get_keywords(pos, without_tag=True)
print(keywords_T)


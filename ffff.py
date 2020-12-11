import os
import re

import numpy as np

import time

from scipy.io.wavfile import write


print("aaa")
labels = np.array([[1, 3, 5], [2, 4, 6], [0, 0, 0]])
print(labels[None, ...])
print(labels[None, :])

_whitespace_re = re.compile(r"\s+")
print(re.sub(_whitespace_re, " ", "nihao ma  wo hen   hao"))

text = "nihaoma wo {B IY}henhao a "
print(len(text))
_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")
print(_curly_re.match(text))

from g2p_en import g2p

g2p = g2p.G2p()
print(g2p('Ｂ'))
# print("==========="*2)
test = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
# for i in test:
#     print(g2p(i))

print(g2p('AE'))
print(g2p('H'))
print(g2p('fetch'))
print(g2p('cake'))
print(g2p('age'))
print(g2p('banana'))

texts = ""
with open("/home/wangyl/test.txt") as text_file:
    texts += " ".join([line.strip() for line in text_file.readlines()])
print(texts)

path = os.path.join("/home/wangyl", "test.txt")
print(str(path))

list_data = ["1", "23", "456", "7890"]
file=open('data.txt','w')
# file.write(str(list_data))
# file.close()

for data in list_data:
    file.write(data)
    file.write('\n')
file.close()

# table = {ord(f): ord(t) for f, t in zip(
#     u'，。！？【】（）％＃＠＆１２３４５６７８９０',
#     u',.!?[]()%#@&1234567890')}
# t = u'中国，中文，标点符号！你好？１２３４５＠＃【】+=-（）'
# t2 = t.translate(table)
# print(t2)


# import unicodedata
#
# t = u'中国，中文，标点符号！你好？Ｂ１２３４５＠＃【】+=-（）'
# t2 = unicodedata.normalize('NFKC', t)
# print(t2)


zh_pattern = re.compile("[\u4e00-\u9fa5]")
mark = {"en": 1, "zh": 2}


# def is_zh(word):
#     global zh_pattern
#     match = zh_pattern.search(word)
#     return match is not None
# http://liyanrui.is-programmer.com/posts/3163.html
def is_zh(c):
    x = ord(c)
    # Punct & Radicals
    if x >= 0x2e80 and x <= 0x33ff:
        return True

    # Fullwidth Latin Characters
    elif x >= 0xff00 and x <= 0xffef:
        return True

    # CJK Unified Ideographs &
    # CJK Unified Ideographs Extension A
    elif x >= 0x4e00 and x <= 0x9fbb:
        return True
    # CJK Compatibility Ideographs
    elif x >= 0xf900 and x <= 0xfad9:
        return True

    # CJK Unified Ideographs Extension B
    elif x >= 0x20000 and x <= 0x2a6d6:
        return True

    # CJK Compatibility Supplement
    elif x >= 0x2f800 and x <= 0x2fa1d:
        return True

    else:
        return False


def split_zh_en(zh_en_str):
    zh_en_group = []
    zh_gather = ""
    en_gather = ""
    zh_status = False

    for c in zh_en_str:
        if not zh_status and is_zh(c):
            zh_status = True
            if en_gather != "":
                zh_en_group.append([mark["en"], en_gather])
                en_gather = ""
        elif not is_zh(c) and zh_status:
            zh_status = False
            if zh_gather != "":
                zh_en_group.append([mark["zh"], zh_gather])
        if zh_status:
            zh_gather += c
        else:
            en_gather += c
            zh_gather = ""

    if en_gather != "":
        zh_en_group.append([mark["en"], en_gather])
    elif zh_gather != "":
        zh_en_group.append([mark["zh"], zh_gather])

    return zh_en_group


if __name__ == '__main__':
    # # tts_model = TTSModel()
    # # text = "我们本次谈话内容录音备案记录，如果您可以在规定时间内下载来分期 艾普 处理逾期账单，我会帮您申请恢复额度，额度恢复以后如果您还有资金需求可以在来分期 艾普 上申请下单，系统审核通过以后，可以再把额度周转出来使用，但若与您协商却无法按时处理，造成的负面影响需自行承担，请提前告知他们有关去电事宜，再见"
    # # print("text>>>>", text)
    # # start_time = time.time()
    # # mels, alignment_history, audios = tts_model.do_synthesis(text)
    # # print("time>>>>>>>", time.time() - start_time)
    # # # print( "audios>>>>", audios  )
    # # # ipd.Audio(audios, rate=24000)
    # # write('/home/wangyl/test.wav', 24000, audios)
    # # # librosa.ex.write_wav("output_seq.wav", audios, 24000)
    # a,b,c = "1 2 3".strip().split()
    # print(a)
    # print(b)
    print(is_zh("。"))
    # print('word。'.encode('UTF-8').isalpha())
    # list_1 = ['a','b']
    # list_1 += ['c','d']
    # print(list_1)
    # print(" ".join(list_1))

    # print("fafsdf".split("-")[0])
    # print(10//3)
    # print(11//3)
    # print(1+15-2%1)
    # ttt = (1,2) * 2
    # print(ttt)
    # print(ttt[:0]+ (-1,))
    # print(ttt[:1]+ttt[2::])

    # str = "2. Good morning / afternoon / evening! 早晨 （下午/晚上）好！"
    # result = split_zh_en(str)
    # print(result)

    print(g2p("hello, world-"))

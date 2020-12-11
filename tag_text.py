import re

zh_pattern = re.compile("[\u4e00-\u9fa5]")


def is_zh(word):
    global zh_pattern
    match = zh_pattern.search(word)
    return match is not None


def tag():
    f1 = open("baker_ali_aicheng/ProsodyLabeling/corpus.txt")
    f2 = open('000000.txt', 'a+')
    lines = f1.readlines()
    result_0, result_1 = [], []
    for idx in range(0, len(lines), 2):
        chn_id, chn_char = lines[idx].strip().split()
        pho_char = lines[idx + 1].strip()
        char_len = len(chn_char)
        i, j = 0, 0
        res_char, pho_str = "", ""
        while i < char_len:
            cur_char = chn_char[i]
            if "，" == cur_char:
                res_char += "#3" + cur_char
            # elif cur_char.encode('UTF-8').isalpha():
            #     res_char += cur_char + "#1"
            elif is_zh(cur_char):
                if cur_char == "不":
                    res_char = res_char + "#1" + cur_char
                elif cur_char == "。":
                    res_char = "#4" + cur_char
                elif cur_char == "是" or cur_char == "经":
                    res_char += cur_char + "#1"
                elif cur_char == "了":
                    res_char += cur_char + "#2"
                elif cur_char == "说" and chn_char[i - 1] == "我":
                    res_char += cur_char + "#1"
                else:
                    res_char += cur_char
            else:
                res_char += cur_char
            i += 1
        a = chn_id + "	" + res_char
        f2.write(a + '\n')
        result_0.append(a)

        pho_len = len(pho_char)
        while j < pho_len:
            cur_char = pho_char[j]
            if cur_char.isupper():
                pho_str += cur_char
            else:
                pho_str += cur_char
            j += 1
        b = "	" + pho_str
        f2.write(b + '\n')
        result_1.append(b)
    return result_0, result_1


if __name__ == '__main__':
    char_0, char_1 = tag()

    # print("A".encode('UTF-8').isalpha())
    # test = "你号码"
    # print(test[:-1]+"吗")

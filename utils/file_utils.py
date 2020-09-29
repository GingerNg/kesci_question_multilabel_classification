import pickle

def readFile(path):
    with open(path, 'r', errors='ignore', encoding="gbk") as file:  # 文档中编码有些问题，所有用errors过滤错误
        content = file.read()
        return content


def writeData(sentences, fileName):
    """
    函数说明：把处理好的写入到文件中，备用
    参数说明：
    """
    out = open(fileName, 'w')
    for sentence in sentences:
        out.write(sentence+"\n")
    print("done!")


def readBunch(path):
    with open(path, 'rb') as file:
        bunch = pickle.load(file)
        # pickle.load(file)
        # 函数的功能：将file中的对象序列化读出。
    return bunch


def writeBunch(path, bunchFile):
    with open(path, 'wb') as file:
        pickle.dump(bunchFile, file)

import jieba
import jieba.analyse
import jieba.posseg as pseg

allow_poses = ('ns', 'n', 'nz','nrt','vn', 'v','t','tg','tg','qt','mq','m','nr','nt',"vn",'q','l')
# 关键词提取：基于TF-IDF:
def get_keywords(content):
    """
    获取关键词
    去停用词
    """
    keywords = jieba.analyse.extract_tags(
                                    sentence=content,
                                    topK=20,   
                                    withWeight=True,
                                    allowPOS=allow_poses
                                    )
    return keywords

if __name__ == "__main__":
    keywords = get_keywords("第9条 合同生效、变更和解除")
    print(keywords)
import jieba
import jieba.analyse
import jieba.posseg as pseg

allow_poses = ('ns', 'n', 'nz', 'nrt', 'vn', 'v', 't', 'tg',
               'tg', 'qt', 'mq', 'm', 'nr', 'nt', "vn", 'q', 'l')
# 关键词提取：基于TF-IDF:


def get_keywords(content, mtd="textrank"):
    """
    获取关键词
    去停用词
    """
    if mtd == "textrank":
        keywords = jieba.analyse.textrank(
            sentence=content,
            topK=20,
            withWeight=True,
            allowPOS=allow_poses
        )
    else:
        keywords = jieba.analyse.extract_tags(
            sentence=content,
            topK=20,
            withWeight=True,
            allowPOS=allow_poses
        )
    return keywords


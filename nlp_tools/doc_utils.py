

# build dataset
def sentence_split(text, vocab, max_sent_len=256, max_segment=16, sent_sep=None):
    """[将一个文本分成多个句子]

    Args:
        text ([type]): [description]
        vocab ([type]): [description]
        max_sent_len (int, optional): [description]. Defaults to 256.
        max_segment (int, optional): [description]. Defaults to 16.

    Returns:
        [type]: [[句子长度， 词list]]
    """
    if sent_sep is None:
        words = text.strip().split()
        document_len = len(words)

        index = list(range(0, document_len, max_sent_len))
        index.append(document_len)

        segments = []
        for i in range(len(index) - 1):
            segment = words[index[i]: index[i + 1]]
            assert len(segment) > 0
            segment = [word if word in vocab._id2word else '<UNK>' for word in segment]
            segments.append([len(segment), segment])
    else:
        segments = []
        sents = text.split(sent_sep)
        for sent in sents:
            segment = sent.split()
            segment = [word if word in vocab._id2word else '<UNK>' for word in segment]
            segments.append(segment)

    assert len(segments) > 0
    # 截断
    if len(segments) > max_segment:
        segment_ = int(max_segment / 2)
        return segments[:segment_] + segments[-segment_:]  # 前后各截1/2
    else:
        return segments

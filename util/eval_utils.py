from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, ngrams, brevity_penalty
from transformers import MBart50Tokenizer


def bleu_upto(reference, hypothesis, n_gram):
    res = []
    for i in range(1, n_gram + 1):
        res.append(calc_bleu_ngram(reference, hypothesis, i))
    return res


def calc_bleu_ngram(reference, hypothesis, n_gram):
    score = 0.0
    ratio = 1 / n_gram

    cc = SmoothingFunction()

    for refer, hypo in zip(reference, hypothesis):
        # refer.index()
        score += sentence_bleu([refer], hypo, (ratio,) * n_gram, cc.method1)

    return score / len(reference)


def remove_padding(decoded, eos_index=2):
    res = []
    for decode in decoded:
        if eos_index in decode:
            eos = decode.index(eos_index)
            decode = decode[:eos]

        res.append(decode)

    return res


def corpuswise_bleu(tokenized_predict, tokenized_gt, tokenizer: MBart50Tokenizer, trg="ko"):
    tokenized_predict = remove_padding(tokenized_predict, 2)  # remove paddings
    tokenized_gt = remove_padding(tokenized_gt, 2)

    detokenized_predict = [tokenizer.decode(predict) for predict in tokenized_predict]
    detokenized_gt = [tokenizer.decode(gt) for gt in tokenized_gt]

    res_predict = []
    res_gt = []

    if trg == "ko":
        import mecab
        mecab = mecab.MeCab()

        for predict in detokenized_predict:
            res_predict.append([i[0] for i in mecab.pos(predict)])

        for gt in detokenized_gt:
            res_gt.append([i[0] for i in mecab.pos(gt)])

        return bleu_upto(res_gt,res_predict,3)

    elif trg=="ja":
        import MeCab
        wakati = MeCab.Tagger("-Owakati")
        for predict in detokenized_predict:
            res_predict.extend([wakati.parse(predict).split()])

        for gt in detokenized_gt:
            res_gt.extend([wakati.parse(gt).split()])

        return bleu_upto(res_gt, res_predict, 3)

    else:
        from sacrebleu import BLEU
        bleu=BLEU()

        return bleu.corpus_score(detokenized_predict, [detokenized_gt],)

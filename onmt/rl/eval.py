from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


def cal_reward(preds, golden):
    infer = [x[0].split() for x in preds]
    golden = [[x] for x in golden]
    nltk_bleu = []
    chencherry = SmoothingFunction()
    nltk_bleu.append(corpus_bleu(golden, infer, smoothing_function=chencherry.method1))
    # for i in range(4):
    #     weights = [1 / (i + 1)] * (i + 1)
    #     nltk_bleu.append(
    #         round(corpus_bleu(
    #             golden, infer, weights=weights, smoothing_function=chencherry.method1), 6))
    return {"bleu": nltk_bleu, "sum_bleu": sum(nltk_bleu)}

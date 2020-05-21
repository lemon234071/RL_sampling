from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


def cal_reward(preds, golden):
    infer = [x[0].split() for x in preds]
    golden = [[x] for x in golden]
    nltk_bleu = []
    chencherry = SmoothingFunction()
    nltk_bleu.append(round(corpus_bleu(golden, infer, smoothing_function=chencherry.method1), 6))
    dist1, dist2 = eval_distinct(infer)
    # for i in range(4):
    #     weights = [1 / (i + 1)] * (i + 1)
    #     nltk_bleu.append(
    #         round(corpus_bleu(
    #             golden, infer, weights=weights, smoothing_function=chencherry.method1), 6))
    return {"bleu": nltk_bleu[0], "dist": round(dist2, 6)}


def get_metric_tokens(infer, golden, bl=False, eval=False):
    chencherry = SmoothingFunction()

    if eval:
        chencherry = SmoothingFunction()
        bleu = round(corpus_bleu(golden, infer, weights=[0.5, 0.5], smoothing_function=chencherry.method1), 7)
        if bl:
            infer = [x[0] for x in golden]
        dist1, dist2 = [round(x, 7) for x in eval_distinct(infer)]
        return {"bleu": bleu, "dist": dist2}

    bleu = []
    dist = []
    for gt, hypts in zip(golden, infer):
        dist.append(eval_distinct(hypts)[1])
        sample_bleu = 0
        for hypt in hypts:
            sample_bleu += corpus_bleu([gt], [hypt], weights=[0.5, 0.5], smoothing_function=chencherry.method1)
        bleu.append(sample_bleu)
    return {"bleu": sum(bleu) / len(bleu), "dist": sum(dist) / len(dist), "bleu_batch": bleu, "dist_batch": dist}


def eval_distinct(hyps_resp):
    """
    compute distinct score for the hyps_resp
    :param hyps_resp: list, a list of hyps responses
    :return: average distinct score for 1, 2-gram
    """
    if len(hyps_resp) == 0:
        print("ERROR, eval_distinct get empty input")
        return

    if type(hyps_resp[0]) != list:
        print("ERROR, eval_distinct takes in a list of <class 'list'>, get a list of {} instead".format(
            type(hyps_resp[0])))
        return

    hyps_resp = [(' '.join(i)).split() for i in hyps_resp]
    # num_tokens = sum([len(i) for i in hyps_resp])
    dist1 = count_ngram_avg(hyps_resp, 1)
    dist2 = count_ngram_avg(hyps_resp, 2)

    return [dist1, dist2]


def count_ngram_avg(hyps_resp, n):
    """
    Count the number of unique n-grams
    :param hyps_resp: list, a list of responses
    :param n: int, n-gram
    :return: the number of unique n-grams in hyps_resp
    """
    if len(hyps_resp) == 0:
        print("ERROR, eval_distinct get empty input")
        return

    if type(hyps_resp[0]) != list:
        print("ERROR, eval_distinct takes in a list of <class 'list'>, get a list of {} instead".format(
            type(hyps_resp[0])))
        return

    tokens = 0
    ngram = set()
    for resp in hyps_resp:
        if len(resp) < n:
            continue
        for i in range(len(resp) - n + 1):
            ngram.add(' '.join(resp[i: i + n]))
            tokens += 1
    return len(ngram) / tokens

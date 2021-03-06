import torch
import torch.nn.functional as F

from onmt.translate.decode_strategy import DecodeStrategy


def get_topp(logits, top_p):
    probs = logits.exp()
    # Compute cumulative probabilities of sorted tokens
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probabilities = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probabilities > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    sorted_samp_probs = sorted_probs.clone()
    sorted_samp_probs[sorted_indices_to_remove] = 0
    ###
    samp_probs = probs.clone()
    samp_probs.scatter_(1, sorted_indices, sorted_samp_probs)
    samp_logits = samp_probs.log()

    # testa = samp_probs.gt(0).sum(1)
    # sorted_next_indices = sorted_samp_probs.multinomial(1).view(-1, 1)
    # next_tokens = sorted_indices.gather(1, sorted_next_indices)
    # next_logprobs = sorted_samp_probs.gather(1, sorted_next_indices).log()

    # Back to unsorted indices and set them to -infinity
    # temp_logits = logits.clone()
    # temp_probs = probs.clone()
    # for i in range(logits.shape[0]):
    #     indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
    #     temp_probs[i][indices_to_remove] = 0
    #     temp_logits[i][indices_to_remove] = -float('Inf')
    # return logits, None
    return samp_logits, None
    # return next_logprobs, next_tokens


# yida translate
def get_topp1(logits, top_p):
    #probs = logits.exp()
    # Compute cumulative probabilities of sorted tokens
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probabilities > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    sorted_samp_logits = sorted_logits.clone()
    sorted_samp_logits[sorted_indices_to_remove] = -float('Inf')
    ###
    samp_logits = logits.clone()
    samp_logits.scatter_(1, sorted_indices, sorted_samp_logits)

    # sorted_next_indices = sorted_samp_probs.multinomial(1).view(-1, 1)
    # next_tokens = sorted_indices.gather(1, sorted_next_indices)
    # next_logprobs = sorted_samp_probs.gather(1, sorted_next_indices).log()

    # Back to unsorted indices and set them to -infinity
    temp_logits = logits.clone()
    for i in range(logits.shape[0]):
        indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
        temp_logits[i][indices_to_remove] = -float('Inf')
    # return logits, None
    return temp_logits, None


def entropy_guide(pos_entropy):
    max_h = pos_entropy.max().clone()
    min_h = pos_entropy.min().clone()

    # for i in range(pos_entropy.shape[0]):
    #     logits[i] = logits[i] / (0.7 * (pos_entropy[i]) / (max_h - min_h))

    return 0.7 * pos_entropy / (max_h - min_h)


def pos_guide(logits, pos_logits, cross=True):
    import json
    with open("E:/git/coai_project/ost/tool_data/vocab_pos_dict.json", "r", encoding="utf-8") as f:
        vocab_pos = json.load(f)
    vocab_pos = list(vocab_pos.values())
    _, pos_saved_indices = get_topp(pos_logits, top_p=0.4)  # bs*pos_size
    pos_vocab_mask = torch.nn.functional.one_hot(torch.tensor(vocab_pos).cuda()).t()  # pos_size*vocab_size
    logits_mask = pos_saved_indices.mm(pos_vocab_mask.float())  # bs*vocab_size
    if cross:
        _, topp_indices = get_topp(logits, top_p=0.9)
        mask = logits_mask * topp_indices

    else:
        mask = logits_mask
        # for i in range(logits.shape[0]):
        #     logits[i][logits_mask[i].byte()] = logits[i][logits_mask[i].byte()] * 0.5
    logits.masked_fill_(~mask.bool(), -10000)
    logits /= 0.7
    return logits


def freq_guide(logits, tag_logits, mask=True):
    topk_tag_scores, topk_tag_ids = tag_logits.topk(1, dim=-1)
    high = topk_tag_ids.eq(4)
    low = topk_tag_ids.eq(5)
    # num = high.float().sum() / topk_pos_ids.shape[0]
    # print(num.item())
    # umerator = high.float() * 0.6 + low.float() * 1.1
    # logits /= numerator
    if mask:
        high_mask = high.squeeze()
        low_mask = low.squeeze()
        index = int(0.003 * (logits.shape[-1] - 4)) + 4
        logits[high_mask, index:] = -10000
        logits[low_mask, : index] = -10000
        # if False:
        #     _, topp_indices = get_topp(logits_backup, top_p=0.9999)
        #     logits.masked_fill_(~topp_indices.bool(), -float('Inf'))
    return logits


def freq_guide_stopwords(logits, pos_logits, mask=True):
    topk_pos_scores, topk_pos_ids = pos_logits.topk(1, dim=-1)
    stop_words = topk_pos_ids.eq(4)
    high = topk_pos_ids.eq(5)
    low = topk_pos_ids.eq(6)
    four = topk_pos_ids.lt(4)
    # low * 0.7 affected eos prob makes seq longer
    numerator = stop_words.float() * 0.01 + high.float() * 0.01 + low.float() * 0.7 + four * 0.7
    logits /= numerator
    if mask:
        import json
        with open("E:/git/coai_project/ost/tool_data/stof.json", "r", encoding="utf-8") as f:
            stof = json.load(f)
        stof = torch.tensor([x for x in stof.values()]).cuda()

        stopwords_mask = stof.eq(4)
        stopwords_mask[:4] = True
        high_mask = stof.eq(5)
        high_mask[:4] = True
        low_mask = stof.eq(6)
        low_mask[:4] = True

        # logits[stop_words.squeeze(), ~stopwords_mask] = -float('Inf')
        # logits[high.squeeze(), ~high_mask] = -float('Inf')
        # logits[low.squeeze(), ~low_mask] = -float('Inf')
        for i in range(logits.shape[0]):
            if stop_words[i][0]:
                logits[i, ~stopwords_mask] = -float('Inf')
            elif high[i][0]:
                logits[i, ~high_mask] = -float('Inf')
            else:
                logits[i, ~low_mask] = -float('Inf')

        # mask_stop = stop_words.float().mm((~stopwords_mask).float().reshape(1, -1))
        # logits = logits.masked_fill_(mask_stop.bool(), -float('Inf'))
        # mask_high = high.float().mm((~high_mask).float().reshape(1, -1))
        # logits = logits.masked_fill_(mask_high.bool(), -float('Inf'))
        # mask_low = low.float().mm((~low_mask).float().reshape(1, -1))
        # logits = logits.masked_fill_(mask_low.bool(), -float('Inf'))
    return logits


# yida translate
def sample_with_dynamic_temperature(logits, tag_logits, sample_method="greedy"):
    if sample_method == "greedy":
        topk_scores, topk_ids = logits.topk(1, dim=-1)
    # elif sample_method == "topp":
    #     topk_scores, topk_ids = get_topp(logits, top_p=0.9)
    else:
        if sample_method == "random":
            logits = logits
        elif sample_method == "topp":
            logits, _ = get_topp(logits, top_p=0.9)
            # logits = logits / 0.7
        # entropy
        # logits = pos_guide(logits, pos_logits)

        ## freq x
        elif sample_method == "freq":
            logits = freq_guide(logits, tag_logits)
        elif sample_method == "topk":
            logits = topk_guide(logits, tag_logits)
        else:
            raise Exception("wrong sample method")

        dist = torch.distributions.Multinomial(
            logits=logits, total_count=1)
        topk_ids = torch.argmax(dist.sample(), dim=1, keepdim=True)
        topk_scores = logits.gather(dim=1, index=topk_ids)

    return topk_ids, topk_scores


def sample_with_temperature(logits, sampling_temp, keep_topk):
    """Select next tokens randomly from the top k possible next tokens.

    Samples from a categorical distribution over the ``keep_topk`` words using
    the category probabilities ``logits / sampling_temp``.

    Args:
        logits (FloatTensor): Shaped ``(batch_size, vocab_size)``.
            These can be logits (``(-inf, inf)``) or log-probs (``(-inf, 0]``).
            (The distribution actually uses the log-probabilities
            ``logits - logits.logsumexp(-1)``, which equals the logits if
            they are log-probabilities summing to 1.)
        sampling_temp (float): Used to scale down logits. The higher the
            value, the more likely it is that a non-max word will be
            sampled.
        keep_topk (int): This many words could potentially be chosen. The
            other logits are set to have probability 0.

    Returns:
        (LongTensor, FloatTensor):

        * topk_ids: Shaped ``(batch_size, 1)``. These are
          the sampled word indices in the output vocab.
        * topk_scores: Shaped ``(batch_size, 1)``. These
          are essentially ``(logits / sampling_temp)[topk_ids]``.
    """

    if sampling_temp == 0.0 or keep_topk == 1:
        # For temp=0.0, take the argmax to avoid divide-by-zero errors.
        # keep_topk=1 is also equivalent to argmax.
        topk_scores, topk_ids = logits.topk(1, dim=-1)
        if sampling_temp > 0:
            topk_scores /= sampling_temp
    else:
        logits = torch.div(logits, sampling_temp)

        if keep_topk > 0:
            top_values, top_indices = torch.topk(logits, keep_topk, dim=1)
            kth_best = top_values[:, -1].view([-1, 1])
            kth_best = kth_best.repeat([1, logits.shape[1]]).float()

            # Set all logits that are not in the top-k to -10000.
            # This puts the probabilities close to 0.
            ignore = torch.lt(logits, kth_best)
            logits = logits.masked_fill(ignore, -10000)

        dist = torch.distributions.Multinomial(
            logits=logits, total_count=1)
        topk_ids = torch.argmax(dist.sample(), dim=1, keepdim=True)
        topk_scores = logits.gather(dim=1, index=topk_ids)
    return topk_ids, topk_scores


class RandomSampling(DecodeStrategy):
    """Select next tokens randomly from the top k possible next tokens.

    The ``scores`` attribute's lists are the score, after applying temperature,
    of the final prediction (either EOS or the final token in the event
    that ``max_length`` is reached)

    Args:
        pad (int): See base.
        bos (int): See base.
        eos (int): See base.
        batch_size (int): See base.
        device (torch.device or str): See base ``device``.
        min_length (int): See base.
        max_length (int): See base.
        block_ngram_repeat (int): See base.
        exclusion_tokens (set[int]): See base.
        return_attention (bool): See base.
        max_length (int): See base.
        sampling_temp (float): See
            :func:`~onmt.translate.random_sampling.sample_with_temperature()`.
        keep_topk (int): See
            :func:`~onmt.translate.random_sampling.sample_with_temperature()`.
        memory_length (LongTensor): Lengths of encodings. Used for
            masking attention.
    """

    def __init__(self, pad, bos, eos, batch_size, device,
                 min_length, block_ngram_repeat, exclusion_tokens,
                 return_attention, max_length, sampling_temp, keep_topk,
                 memory_length,
                 # yida translate
                 tag_gen, sample_method, tag_src):
        super(RandomSampling, self).__init__(
            pad, bos, eos, batch_size, device, 1,
            min_length, block_ngram_repeat, exclusion_tokens,
            return_attention, max_length)
        self.sampling_temp = sampling_temp
        self.keep_topk = keep_topk
        self.topk_scores = None
        self.memory_length = memory_length
        self.batch_size = batch_size
        self.select_indices = torch.arange(self.batch_size,
                                           dtype=torch.long, device=device)
        self.original_batch_idx = torch.arange(self.batch_size,
                                               dtype=torch.long, device=device)
        # yida translate
        self.tag_predictions = [[] for _ in range(batch_size)]
        self.pos_alive_seq = torch.full(
            [batch_size * 1, 1], bos,
            dtype=torch.long, device=device)
        self.tag_gen = tag_gen

        self.sample_method = sample_method
        self.tag_alive_src = tag_src

    # yida translate
    def advance(self, log_probs, attn, pos_log_probs):
        """Select next tokens randomly from the top k possible next tokens.

        Args:
            log_probs (FloatTensor): Shaped ``(batch_size, vocab_size)``.
                These can be logits (``(-inf, inf)``) or log-probs
                (``(-inf, 0]``). (The distribution actually uses the
                log-probabilities ``logits - logits.logsumexp(-1)``,
                which equals the logits if they are log-probabilities summing
                to 1.)
            attn (FloatTensor): Shaped ``(1, B, inp_seq_len)``.
        """

        self.ensure_min_length(log_probs)
        self.block_ngram_repeats(log_probs)
        # yida translate
        self.ensure_min_length(pos_log_probs)
        self.block_ngram_repeats(pos_log_probs)
        if self.tag_gen:
            _, topk_pos_ids = pos_log_probs.topk(1, dim=-1)
            self.pos_alive_seq = torch.cat([self.pos_alive_seq, topk_pos_ids], -1)

        # # yida translate
        # dynamic = True
        # if not (dynamic and self.pos_gen):
        #     topk_ids, self.topk_scores = sample_with_temperature(
        #         log_probs, self.sampling_temp, self.keep_topk)
        # else:
        #     topk_ids, self.topk_scores = \
        #         sample_with_dynamic_temperature(log_probs, pos_log_probs, self.leanred_t)
        topk_ids, self.topk_scores = \
            sample_with_dynamic_temperature(log_probs, pos_log_probs, self.sample_method)

        self.is_finished = topk_ids.eq(self.eos)

        self.alive_seq = torch.cat([self.alive_seq, topk_ids], -1)
        if self.return_attention:
            if self.alive_attn is None:
                self.alive_attn = attn
            else:
                self.alive_attn = torch.cat([self.alive_attn, attn], 0)
        self.ensure_max_length()

    def update_finished(self):
        """Finalize scores and predictions."""
        # shape: (sum(~ self.is_finished), 1)
        finished_batches = self.is_finished.view(-1).nonzero()
        for b in finished_batches.view(-1):
            b_orig = self.original_batch_idx[b]
            self.scores[b_orig].append(self.topk_scores[b, 0])
            self.predictions[b_orig].append(self.alive_seq[b, 1:])
            if self.tag_gen:
                self.tag_predictions[b_orig].append(self.pos_alive_seq[b, 1:])

            self.attention[b_orig].append(
                self.alive_attn[:, b, :self.memory_length[b]]
                if self.alive_attn is not None else [])
        self.done = self.is_finished.all()
        if self.done:
            return
        is_alive = ~self.is_finished.view(-1)
        self.alive_seq = self.alive_seq[is_alive]
        # yida translate
        if self.tag_gen:
            self.pos_alive_seq = self.pos_alive_seq[is_alive]
        if self.tag_alive_src is not None:
            self.tag_alive_src = self.tag_alive_src[:, is_alive]

        if self.alive_attn is not None:
            self.alive_attn = self.alive_attn[:, is_alive]
        self.select_indices = is_alive.nonzero().view(-1)
        self.original_batch_idx = self.original_batch_idx[is_alive]

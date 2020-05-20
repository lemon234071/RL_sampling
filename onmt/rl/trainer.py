#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function

import codecs
import collections
import math
import os
import time
from itertools import count

import torch
from tensorboardX import SummaryWriter

import onmt.decoders.ensemble
import onmt.inputters as inputters
import onmt.model_builder
import onmt.rl.beam
# yida RL translate
from onmt.rl.eval import get_metric_tokens
from onmt.rl.random_sampling import RandomSampling
from onmt.utils.misc import set_random_seed
from utils import *


def get_topp(logits, top_p=0.9):
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
    return samp_logits


def build_rltor_enc(opt, rl_model, optim, model_saver, report_score=True, logger=None, out_file=None):
    if out_file is None:
        out_file = codecs.open(opt.output, 'w+', 'utf-8')

    load_test_model = onmt.decoders.ensemble.load_test_model \
        if len(opt.models) > 1 else onmt.model_builder.load_test_model
    fields, model, model_opt = load_test_model(opt)

    scorer = onmt.rl.GNMTGlobalScorer.from_opt(opt)

    translator = Translator.from_opt(
        model,
        rl_model, optim, model_saver,
        fields,
        opt,
        model_opt,
        global_scorer=scorer,
        out_file=out_file,
        report_score=report_score,
        logger=logger
    )
    return translator


def max_tok_len(new, count, sofar):
    """
    In token batching scheme, the number of sequences is limited
    such that the total number of src/tgt tokens (including padding)
    in a batch <= batch_size
    """
    # Maintains the longest src and tgt length in the current batch
    global max_src_in_batch  # this is a hack
    # Reset current longest length at a new batch (count=1)
    if count == 1:
        max_src_in_batch = 0
        # max_tgt_in_batch = 0
    # Src: [<bos> w1 ... wN <eos>]
    max_src_in_batch = max(max_src_in_batch, len(new.src[0]) + 2)
    # Tgt: [w1 ... wM <eos>]
    src_elements = count * max_src_in_batch
    return src_elements


class Translator(object):
    """Translate a batch of sentences with a saved model.

    Args:
        model (onmt.modules.NMTModel): NMT model to use for translation
        fields (dict[str, torchtext.data.Field]): A dict
            mapping each side to its list of name-Field pairs.
        src_reader (onmt.inputters.DataReaderBase): Source reader.
        tgt_reader (onmt.inputters.TextDataReader): Target reader.
        gpu (int): GPU device. Set to negative for no GPU.
        n_best (int): How many beams to wait for.
        min_length (int): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        max_length (int): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        beam_size (int): Number of beams.
        random_sampling_topk (int): See
            :class:`onmt.translate.random_sampling.RandomSampling`.
        random_sampling_temp (int): See
            :class:`onmt.translate.random_sampling.RandomSampling`.
        stepwise_penalty (bool): Whether coverage penalty is applied every step
            or not.
        dump_beam (bool): Debugging option.
        block_ngram_repeat (int): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        ignore_when_blocking (set or frozenset): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        replace_unk (bool): Replace unknown token.
        data_type (str): Source data type.
        verbose (bool): Print/log every translation.
        report_bleu (bool): Print/log Bleu metric.
        report_rouge (bool): Print/log Rouge metric.
        report_time (bool): Print/log total time/frequency.
        copy_attn (bool): Use copy attention.
        global_scorer (onmt.translate.GNMTGlobalScorer): Translation
            scoring/reranking object.
        out_file (TextIO or codecs.StreamReaderWriter): Output file.
        report_score (bool) : Whether to report scores
        logger (logging.Logger or NoneType): Logger.
    """

    def __init__(
            self,
            model,
            rl_model, optim, model_saver,
            fields,
            src_reader,
            tgt_reader,
            gpu=-1,
            n_best=1,
            min_length=0,
            max_length=100,
            ratio=0.,
            beam_size=30,
            random_sampling_topk=1,
            random_sampling_temp=1,
            sta=False,
            stepwise_penalty=None,
            dump_beam=False,
            block_ngram_repeat=0,
            ignore_when_blocking=frozenset(),
            replace_unk=False,
            phrase_table="",
            data_type="text",
            verbose=False,
            report_bleu=False,
            report_rouge=False,
            report_time=False,
            copy_attn=False,
            global_scorer=None,
            out_file=None,
            report_score=True,
            logger=None,
            sample_method="freq",
            tag_mask_path="",
            mask_attn=False,
            epochs=4,
            samples_n=2,
            report_every=5,
            valid_steps=30,
            save_checkpoint_steps=100,
            random_steps=10000,
            seed=-1):
        self.model = model
        self.fields = fields
        tgt_field = dict(self.fields)["tgt"].base_field
        self._tgt_vocab = tgt_field.vocab
        self._tgt_eos_idx = self._tgt_vocab.stoi[tgt_field.eos_token]
        self._tgt_pad_idx = self._tgt_vocab.stoi[tgt_field.pad_token]
        self._tgt_bos_idx = self._tgt_vocab.stoi[tgt_field.init_token]
        self._tgt_unk_idx = self._tgt_vocab.stoi[tgt_field.unk_token]
        self._tgt_vocab_len = len(self._tgt_vocab)

        self._gpu = gpu
        self._use_cuda = gpu > -1
        self._dev = torch.device("cuda", self._gpu) \
            if self._use_cuda else torch.device("cpu")

        self.n_best = n_best
        self.max_length = max_length

        self.beam_size = beam_size
        self.random_sampling_temp = random_sampling_temp
        self.sample_from_topk = random_sampling_topk

        self.min_length = min_length
        self.ratio = ratio
        self.stepwise_penalty = stepwise_penalty
        self.dump_beam = dump_beam
        self.block_ngram_repeat = block_ngram_repeat
        self.ignore_when_blocking = ignore_when_blocking
        self._exclusion_idxs = {
            self._tgt_vocab.stoi[t] for t in self.ignore_when_blocking}
        self.src_reader = src_reader
        self.tgt_reader = tgt_reader
        self.replace_unk = replace_unk
        if self.replace_unk and not self.model.decoder.attentional:
            raise ValueError(
                "replace_unk requires an attentional decoder.")
        self.phrase_table = phrase_table
        self.data_type = data_type
        self.verbose = verbose
        self.report_bleu = report_bleu
        self.report_rouge = report_rouge
        self.report_time = report_time

        self.copy_attn = copy_attn

        self.global_scorer = global_scorer
        if self.global_scorer.has_cov_pen and \
                not self.model.decoder.attentional:
            raise ValueError(
                "Coverage penalty requires an attentional decoder.")
        self.out_file = out_file
        self.report_score = report_score
        self.logger = logger

        self.use_filter_pred = False
        self._filter_pred = None

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None
        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

        # for rl
        self.mask_attn = mask_attn
        self.sta = sta
        self.epochs = 1 if sta else epochs
        self.report_every = report_every
        self.valid_steps = valid_steps
        self.save_checkpoint_steps = save_checkpoint_steps
        self.samples_n = samples_n
        self.samples_method = sample_method
        self.writer = SummaryWriter()
        self.rl_model, self.optim, self.model_saver = rl_model, optim, model_saver
        self.criterion = torch.nn.NLLLoss(reduction='none')
        self.random_steps = random_steps
        self.tag_mask = load_json(tag_mask_path)
        for k in self.tag_mask:
            self.tag_mask[k] = torch.tensor(self.tag_mask[k], dtype=torch.float, device=self._dev).unsqueeze(0)
        self.tag_vocab = dict(self.fields)["tag_tgt"].base_field.vocab.stoi

        self.rl_model.train()
        self.model.eval()

        set_random_seed(seed, self._use_cuda)

    @classmethod
    def from_opt(
            cls,
            model,
            rl_model, optim, model_saver,
            fields,
            opt,
            model_opt,
            global_scorer=None,
            out_file=None,
            report_score=True,
            logger=None):
        """Alternate constructor.

        Args:
            model (onmt.modules.NMTModel): See :func:`__init__()`.
            fields (dict[str, torchtext.data.Field]): See
                :func:`__init__()`.
            opt (argparse.Namespace): Command line options
            model_opt (argparse.Namespace): Command line options saved with
                the model checkpoint.
            global_scorer (onmt.translate.GNMTGlobalScorer): See
                :func:`__init__()`..
            out_file (TextIO or codecs.StreamReaderWriter): See
                :func:`__init__()`.
            report_score (bool) : See :func:`__init__()`.
            logger (logging.Logger or NoneType): See :func:`__init__()`.
        """

        src_reader = inputters.str2reader[opt.data_type].from_opt(opt)
        tgt_reader = inputters.str2reader["text"].from_opt(opt)
        return cls(
            model,
            rl_model, optim, model_saver,
            fields,
            src_reader,
            tgt_reader,
            gpu=opt.gpu,
            n_best=opt.n_best,
            min_length=opt.min_length,
            max_length=opt.max_length,
            ratio=opt.ratio,
            beam_size=opt.beam_size,
            random_sampling_topk=opt.random_sampling_topk,
            random_sampling_temp=opt.random_sampling_temp,
            stepwise_penalty=opt.stepwise_penalty,
            dump_beam=opt.dump_beam,
            block_ngram_repeat=opt.block_ngram_repeat,
            ignore_when_blocking=set(opt.ignore_when_blocking),
            replace_unk=opt.replace_unk,
            phrase_table=opt.phrase_table,
            data_type=opt.data_type,
            verbose=opt.verbose,
            report_bleu=opt.report_bleu,
            report_rouge=opt.report_rouge,
            report_time=opt.report_time,
            copy_attn=model_opt.copy_attn,
            global_scorer=global_scorer,
            out_file=out_file,
            report_score=report_score,
            logger=logger,
            # yida
            epochs=opt.epochs,
            report_every=opt.report_every,
            valid_steps=opt.valid_steps,
            save_checkpoint_steps=opt.save_checkpoint_steps,
            random_steps=opt.random_steps,
            samples_n=opt.rl_samples,
            sample_method=opt.sample_method,
            tag_mask_path=opt.tag_mask,
            mask_attn=model_opt.mask_attn,
            sta=opt.statistic,
            #
            seed=opt.seed)

    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def _gold_score(self, batch, memory_bank, src_lengths, src_vocabs,
                    use_src_map, enc_states, batch_size, src):
        gs = [0] * batch_size
        return gs

    def infer(
            self,
            test_src,
            test_tag_src,
            batch_size=None,
            batch_type="sents"):
        if batch_size is None:
            raise ValueError("batch_size must be set")

        test_data = inputters.Dataset(
            self.fields,
            readers=([self.src_reader, self.tgt_reader]
                     if test_tag_src is not None else [self.src_reader]),
            data=[("src", test_src), ("tag_src", test_tag_src)]
            if test_tag_src is not None else [("src", test_src)],
            dirs=[None, None],
            sort_key=inputters.str2sortkey[self.data_type],
            filter_pred=self._filter_pred
        )

        test_iter = inputters.OrderedIterator(
            dataset=test_data,
            device=self._dev,
            batch_size=batch_size,
            batch_size_fn=max_tok_len if batch_type == "tokens" else None,
            train=False,
            sort=False,
            sort_within_batch=True,
            shuffle=False
        )

        xlation_builder = onmt.rl.TranslationBuilder(
            test_data, self.fields, self.n_best, self.replace_unk, False,
            self.phrase_table
        )

        # # Statistics
        counter = count(1)
        pred_score_total, pred_words_total = 0, 0

        all_predictions = []

        start_time = time.time()

        self.validate(test_iter, test_data, xlation_builder, infer=True)
        end_time = time.time()

        if self.report_time:
            total_time = end_time - start_time
            self._log("Total translation time (s): %f" % total_time)
            self._log("Average translation time (s): %f" % (
                    total_time / len(all_predictions)))
            self._log("Tokens per second: %f" % (
                    pred_words_total / total_time))

    def train(
            self,
            train_src,
            train_tgt,
            train_tag_src,
            train_tag_tgt,
            valid_src,
            valid_tgt,
            valid_tag_src,
            valid_tag_tgt,
            src_dir=None,
            batch_size=None,
            batch_type="sents"):
        """Translate content of ``src`` and get gold scores from ``tgt``.

        Args:
            src: See :func:`self.src_reader.read()`.
            tgt: See :func:`self.tgt_reader.read()`.
            src_dir: See :func:`self.src_reader.read()` (only relevant
                for certain types of data).
            batch_size (int): size of examples per mini-batch
            attn_debug (bool): enables the attention logging

        Returns:
            (`list`, `list`)

            * all_scores is a list of `batch_size` lists of `n_best` scores
            * all_predictions is a list of `batch_size` lists
                of `n_best` predictions
        """

        if batch_size is None:
            raise ValueError("batch_size must be set")

        train_data = inputters.Dataset(
            self.fields,
            # yida translate
            readers=([self.src_reader, self.tgt_reader, self.tgt_reader, self.tgt_reader]
                     if train_tag_tgt is not None else [self.src_reader, self.tgt_reader]),
            data=[("src", train_src), ("tgt", train_tgt), ("tag_src", train_tag_src), ("tag_tgt", train_tag_tgt)]
            if train_tag_tgt is not None else [("src", train_src), ("tgt", train_tgt)],
            dirs=[src_dir, None, None, None],
            sort_key=inputters.str2sortkey[self.data_type],
            filter_pred=self._filter_pred
        )

        train_iter = inputters.OrderedIterator(
            dataset=train_data,
            device=self._dev,
            batch_size=batch_size,
            pool_factor=8192,
            batch_size_fn=max_tok_len if batch_type == "tokens" else None,
            train=True,
            sort=False,
            sort_within_batch=True
        )

        valid_data = inputters.Dataset(
            self.fields,
            # yida translate
            readers=([self.src_reader, self.tgt_reader, self.tgt_reader, self.tgt_reader]
                     if train_tag_tgt is not None else [self.src_reader, self.tgt_reader]),
            data=[("src", valid_src), ("tgt", valid_tgt), ("tag_src", valid_tag_src), ("tag_tgt", valid_tag_tgt)]
            if train_tag_tgt is not None else [("src", valid_src), ("tgt", valid_tgt)],
            dirs=[src_dir, None, None, None],
            sort_key=inputters.str2sortkey[self.data_type],
            filter_pred=self._filter_pred
        )

        valid_iter = inputters.OrderedIterator(
            dataset=valid_data,
            device=self._dev,
            batch_size=batch_size,
            pool_factor=8192,
            batch_size_fn=max_tok_len if batch_type == "tokens" else None,
            train=False,
            sort=False,
            sort_within_batch=True
        )

        train_xlation_builder = onmt.rl.TranslationBuilder(
            train_data, self.fields, self.n_best, self.replace_unk, False,
            self.phrase_table
        )

        valid_xlation_builder = onmt.rl.TranslationBuilder(
            valid_data, self.fields, self.n_best, self.replace_unk, False,
            self.phrase_table
        )

        # # Statistics
        counter = count(1)
        pred_score_total, pred_words_total = 0, 0
        gold_score_total, gold_words_total = 0, 0

        all_scores = []
        all_predictions = []

        start_time = time.time()

        for epoch in range(self.epochs):
            for batch in train_iter:
                step = self.optim.training_step

                if step % self.valid_steps == 0 or step == 1:
                    self.validate(valid_iter, valid_data, valid_xlation_builder)

                self._gradient_accumulation(batch, train_data, train_xlation_builder)

                if (self.model_saver is not None
                        and (self.save_checkpoint_steps != 0
                             and step % self.save_checkpoint_steps == 0)):
                    self.model_saver.save(step, moving_average=None)
            print("!!!!!!!!!!!!!")
            print("epoch:", epoch)
            print("!!!!!!!!!!!!!")
        if self.model_saver is not None:
            self.model_saver.save(self.optim.training_step, moving_average=None)

        end_time = time.time()

        if self.report_time:
            total_time = end_time - start_time
            self._log("Total translation time (s): %f" % total_time)
            self._log("Average translation time (s): %f" % (
                    total_time / len(all_predictions)))
            self._log("Tokens per second: %f" % (
                    pred_words_total / total_time))

        return all_scores, all_predictions

    def _gradient_accumulation(self, batch, data, xlation_builder):
        # Encoder forward.
        with torch.no_grad():
            src, enc_states, memory_bank, src_lengths = self._run_encoder(batch)

        self.optim.zero_grad()

        # prepare input
        # input = enc_states.transpose(0, 1)
        # input = input.contiguous().view(input.size(0), -1)
        fix_k = 1
        if self.optim.training_step % 100 == 0:
            fix_k = 1 - fix_k
        if self.optim.training_step > 1200 or self.optim.training_step < 600:
            fix_k = None
        fix_k = None

        inputs = enc_states[-1].squeeze()
        log_probs = self.rl_model(inputs, fix_k=str(fix_k))
        # log_probs["1"].detach()

        # compute loss
        loss = self._compute_loss_k(log_probs, batch, data, xlation_builder,
                                    src, enc_states, memory_bank, src_lengths)
        # loss.backward()
        if loss is not None:
            self.optim.backward(loss)
        self.optim.step()

    def _compute_loss_k(self, log_probs, batch, data, xlation_builder, src, enc_states, memory_bank,
                        src_lengths):

        # sampling temperature
        ts_ids = {}
        for k, logits in log_probs.items():
            dist = torch.distributions.Multinomial(logits=logits, total_count=1)
            ts_ids[k] = [torch.argmax(dist.sample(), dim=1, keepdim=True) for i in range(self.samples_n)]
        loss_t = torch.tensor([0] * self.samples_n, dtype=torch.float, device=self._dev)
        learned_t = {}
        for k, logits_t in log_probs.items():
            learned_t[k] = self.tid2t(ts_ids[k], k)
            tids = torch.stack(ts_ids[k], -1)
            # k_topk_ids = torch.cat(k_topk_ids, 0)
            k_logits_t = torch.stack([logits_t] * self.samples_n, -1)
            # k_logits_t = torch.cat([logits_t] * k, 0)
            loss_t += self.criterion(k_logits_t, tids.squeeze()).mean(0)

        # infer samples(slow or not
        attn_debug = False
        k_reward_qs = []
        k_bleu = []
        k_dist = []
        for i in range(self.samples_n):
            with torch.no_grad():
                self.model.decoder.init_state(src, memory_bank, enc_states)
            batch_data = self.translate_batch(
                batch, data.src_vocabs, attn_debug, memory_bank, src_lengths, enc_states, src,
                {k: v[i] for k, v in learned_t.items()}, sample_method=self.samples_method
            )
            pred_ids, gt_ids = self.tensor2ids(batch_data)
            metrices = get_metric_tokens(pred_ids, gt_ids)

            # # # translate, so slow
            # batch_sents, golden_truth = self.ids2sents(batch_data, xlation_builder)
            # # cal rewards
            # reward_dict = cal_reward(batch_sents, golden_truth)
            reward_sample = self._compute_reward(metrices)
            k_reward_qs.append(reward_sample)
            k_bleu.append(metrices["bleu"])
            k_dist.append(metrices["dist"])

        # bl
        argmax_t = {}
        for k, logits_t in log_probs.items():
            _, topk_ids = logits_t.topk(1, dim=-1)
            argmax_t[k] = self.tid2t([topk_ids], k)[0]
        with torch.no_grad():
            self.model.decoder.init_state(src, memory_bank, enc_states)
        bl_batch_data = self.translate_batch(
            batch, data.src_vocabs, attn_debug, memory_bank, src_lengths, enc_states, src,
            argmax_t, sample_method="greedy"
        )
        arg_pred_ids, arg_gt_ids = self.tensor2ids(bl_batch_data)
        metrices_bl = get_metric_tokens(arg_pred_ids, arg_gt_ids)
        reward_argmax = self._compute_reward(metrices_bl)

        reward_mean = sum(k_reward_qs) / len(k_reward_qs)
        # reward_bl = reward_mean
        # reward_bl = reward_argmax
        # reward = (torch.tensor(k_reward_qs, device=self._dev) - reward_bl) / max(
        #     [abs(x - reward_bl) + 1e-8 for x in k_reward_qs])
        reward = (torch.tensor(k_reward_qs, device=self._dev) - min(k_reward_qs)) \
                 / (max(k_reward_qs) - min(k_reward_qs) + 1e-8)
        loss = reward * loss_t

        # sta
        if self.optim.training_step % self.report_every == 0:
            for k, v in learned_t.items():
                self.writer.add_scalars(
                    "train_t/{}".format(k),
                    {"t_mean": v.mean(), "mean+stc": v.mean() + v.std(), "mean-stc": v.mean() - v.std()},
                    self.optim.training_step)
            self.writer.add_scalars("train_loss", {"loss_mean": loss_t.mean()}, self.optim.training_step)
            self.writer.add_scalars("train_reward/reward", {"argmax": reward_argmax, "mean": reward_mean},
                                    self.optim.training_step)
            self.writer.add_scalars("train_reward/bleu",
                                    {"argmax": metrices_bl["bleu"], "mean": sum(k_bleu) / len(k_bleu)},
                                    self.optim.training_step)
            self.writer.add_scalars("train_reward/dist",
                                    {"argmax": metrices_bl["dist"], "mean": sum(k_dist) / len(k_dist)},
                                    self.optim.training_step)
            self.writer.add_scalars("lr", {"lr": self.optim.learning_rate()}, self.optim.training_step)
        return loss

    def _compute_reward(self, metric_dict):
        reward = metric_dict["bleu"] + metric_dict["dist"]
        return reward

    def tid2t(self, t_ids, k):
        # return t_ids.float() + 0.001
        t = (torch.stack(t_ids, 0).float() + 1)
        if k == "low":
            t += 5
        return t / 10

    def tensor2ids(self, batch_data):
        batch = batch_data["batch"]
        predictions = []
        for instance in batch_data["predictions"]:
            seq = instance[0].tolist()
            if seq[-1] == self._tgt_eos_idx:
                seq = seq[:-1]
            predictions.append([str(x) for x in seq])
        # predictions = [[str(x) for x in instance[0].tolist()] if for instance in batch_data["predictions"]]
        mask = batch.tgt.T.ne(self._tgt_pad_idx)
        gt_tokens = [[[str(x) for x in batch.tgt.T[0, i][mask[0, i]][1:-1].tolist()]] for i in range(batch.batch_size)]
        return predictions, gt_tokens

    def validate(self, valid_iter, data, xlation_builder, infer=False, attn_debug=False):
        loss_total = 0.0
        step = 0
        all_predictions = []
        arg_prediction = []
        golden = []
        t_all = collections.defaultdict(list)
        t_arg_all = collections.defaultdict(list)

        self.rl_model.eval()
        with torch.no_grad():
            for batch in valid_iter:
                src, enc_states, memory_bank, src_lengths = self._run_encoder(batch)
                self.model.decoder.init_state(src, memory_bank, enc_states)
                # input = enc_states.transpose(0, 1)
                # input = input.contiguous().view(input.size(0), -1)
                inputs = enc_states[-1].squeeze()

                # F-prop through the model.
                log_probs = self.rl_model(inputs)
                if not (infer or self.sta):
                    learned_t = {}
                    loss_t = torch.tensor(0, dtype=torch.float, device=self._dev)
                    for k, logits_t in log_probs.items():
                        dist = torch.distributions.Multinomial(logits=logits_t, total_count=1)
                        k_topk_ids = torch.argmax(dist.sample(), dim=1, keepdim=True)
                        learned_t[k] = self.tid2t([k_topk_ids], k)[0]
                        loss_t += self.criterion(logits_t, k_topk_ids.view(-1)).mean()

                    batch_data = self.translate_batch(
                        batch, data.src_vocabs, attn_debug, memory_bank, src_lengths, enc_states, src,
                        learned_t, sample_method=self.samples_method
                    )
                    batch_sents, golden_truth = self.tensor2ids(batch_data)
                    # batch_sents, golden_truth = self.ids2sents(batch_data, xlation_builder)
                    all_predictions += batch_sents
                    golden += golden_truth

                    loss_total += loss_t
                    for k in learned_t.keys():
                        t_all[k].append(learned_t[k])

                # arg
                learned_t_arg = {}
                for k, logits_t in log_probs.items():
                    _, topk_ids = logits_t.topk(1, dim=-1)
                    learned_t_arg[k] = self.tid2t([topk_ids], k)[0]
                    # loss_t += self.criterion(logits_t, topk_ids.view(-1))

                self.model.decoder.init_state(src, memory_bank, enc_states)
                batch_data_arg = self.translate_batch(
                    batch, data.src_vocabs, attn_debug, memory_bank, src_lengths, enc_states, src,
                    learned_t_arg, sample_method=self.samples_method, sta=self.sta)
                for k in learned_t_arg.keys():
                    t_arg_all[k].append(learned_t_arg[k])
                # translate, so slow
                if not self.sta:
                    if infer:
                        batch_sents_arg, _ = self.ids2sents(batch_data_arg, xlation_builder, infer)
                    else:
                        batch_sents_arg, _ = self.tensor2ids(batch_data_arg)
                    arg_prediction += batch_sents_arg
                step += 1
        if infer:
            for k, v in t_arg_all.items():
                v = torch.cat(v)
                print("     arg t {}:".format(k), v.mean(), "std:", v.std(), self.optim.training_step)
            return arg_prediction
        # diff unk
        metirc_argmax = get_metric_tokens(arg_prediction, golden)
        metirc_sample = get_metric_tokens(all_predictions, golden)
        # metirc_argmax = cal_reward(arg_prediction, golden)
        # metirc_sample = cal_reward(all_predictions, golden)
        reward = self._compute_reward(metirc_sample)
        reward_arg = self._compute_reward(metirc_argmax)
        self.writer.add_scalars("valid_loss", {"loss": loss_total / step}, self.optim.training_step)
        self.writer.add_scalars("valid_reward/reward",
                                {"reward_sample": reward, "reward_arg": reward_arg}, self.optim.training_step)
        self.writer.add_scalars("valid_reward/bleu",
                                {"bleu": metirc_sample["bleu"],
                                 "bleu_arg": metirc_argmax["bleu"]}, self.optim.training_step)
        self.writer.add_scalars("valid_reward/dist",
                                {"dist": metirc_sample["dist"],
                                 "dist_arg": metirc_argmax["dist"]}, self.optim.training_step)
        for k, v in t_all.items():
            v = torch.cat(v)
            self.writer.add_scalars(
                "valid_t/sample_{}".format(k),
                {"t_mean": v.mean(), "mean+stc": v.mean() + v.std(), "mean-stc": v.mean() - v.std()},
                self.optim.training_step)
        for k, v in t_arg_all.items():
            v = torch.cat(v)
            self.writer.add_scalars(
                "valid_t/arg_{}".format(k),
                {"t_mean": v.mean(), "mean+stc": v.mean() + v.std(), "mean-stc": v.mean() - v.std()},
                self.optim.training_step)
            print("     arg t {}:".format(k), v.mean(), "std:", v.std(), self.optim.training_step)
        print("valid loss:", loss_total / step)
        self.rl_model.train()

    def ids2sents(
            self,
            batch_data,
            xlation_builder,
            infer=False):
        batch_predictions = []
        batch_gt = []
        translations = xlation_builder.from_batch(batch_data)
        for trans in translations:

            n_best_preds = [" ".join(pred)
                            for pred in trans.pred_sents[:self.n_best]]
            batch_predictions += [n_best_preds]
            batch_gt += [trans.tgt_raw]
            if infer:
                self.out_file.write('\n'.join(n_best_preds) + '\n')
                self.out_file.flush()
        return batch_predictions, batch_gt

    def _translate_random_sampling(
            self,
            batch,
            src_vocabs,
            max_length,
            # yida RL translate
            memory_bank, src_lengths, enc_states, src, learned_t, sta, sample_method,
            #
            min_length=0,
            sampling_temp=1.0,
            keep_topk=-1,
            return_attention=False,
            # yida RL translate
            vocab_pos=None):
        """Alternative to beam search. Do random sampling at each step."""

        assert self.beam_size == 1

        # TODO: support these blacklisted features.
        assert self.block_ngram_repeat == 0

        batch_size = batch.batch_size

        # # Encoder forward.
        # src, enc_states, memory_bank, src_lengths = self._run_encoder(batch)
        # self.model.decoder.init_state(src, memory_bank, enc_states)

        use_src_map = self.copy_attn

        results = {
            "predictions": None,
            "scores": None,
            "attention": None,
            "batch": batch,
            "gold_score": self._gold_score(
                batch, memory_bank, src_lengths, src_vocabs, use_src_map,
                enc_states, batch_size, src)}

        memory_lengths = src_lengths
        src_map = batch.src_map if use_src_map else None

        if isinstance(memory_bank, tuple):
            mb_device = memory_bank[0].device
        else:
            mb_device = memory_bank.device

        tag_src = None
        if self.mask_attn:
            tag_src, _ = batch.tag_src if isinstance(batch.tag_src, tuple) else (batch.tag_src, None)

        random_sampler = RandomSampling(
            self._tgt_pad_idx, self._tgt_bos_idx, self._tgt_eos_idx,
            batch_size, mb_device, min_length, self.block_ngram_repeat,
            self._exclusion_idxs, return_attention, self.max_length,
            sampling_temp, keep_topk, memory_lengths,
            # yida translate
            None, vocab_pos, learned_t, sample_method, tag_src)

        for step in range(max_length):
            # Shape: (1, B, 1)
            decoder_input = random_sampler.alive_seq[:, -1].view(1, -1, 1)
            tag_decoder_src = random_sampler.tag_alive_src \
                if self.model.mask_attn else None
            tag_decoder_in = random_sampler.pos_alive_seq[:, -1].view(1, -1, 1) \
                if "tag" in self.model.generators else None
            t_in = random_sampler.learned_t

            # yida translate
            log_probs, attn, pos_log_probs, pass_indices = self._decode_and_generate(
                decoder_input,
                memory_bank,
                # yida tranlate
                tag_decoder_src,
                tag_decoder_in,
                t_in,
                batch,
                src_vocabs,
                memory_lengths=memory_lengths,
                src_map=src_map,
                step=step,
                batch_offset=random_sampler.select_indices
            )

            # yida tranlate
            random_sampler.advance(log_probs, attn, pos_log_probs, sta, pass_indices)

            # for k, v in pass_indices.items():
            #     k_left = random_sampler.num_topp_left[v].float()
            #     if k_left.shape[0] > 0:
            #         temp_std = k_left.std() if k_left.shape[0] > 1 else 0
            #         self.writer.add_scalars(
            #             "{}_topp_left/position_{}".format(k, step),
            #             {"mean": k_left.mean(),
            #              "mean+std": k_left.mean() + temp_std,
            #              "mean-std": k_left.mean() - temp_std},
            #             self.optim.training_step)

            any_batch_is_finished = random_sampler.is_finished.any()
            if any_batch_is_finished:
                random_sampler.update_finished()
                if random_sampler.done:
                    break

            if any_batch_is_finished:
                select_indices = random_sampler.select_indices

                # Reorder states.
                if isinstance(memory_bank, tuple):
                    memory_bank = tuple(x.index_select(1, select_indices)
                                        for x in memory_bank)
                else:
                    memory_bank = memory_bank.index_select(1, select_indices)

                memory_lengths = memory_lengths.index_select(0, select_indices)

                if src_map is not None:
                    src_map = src_map.index_select(1, select_indices)

                self.model.decoder.map_state(
                    lambda state, dim: state.index_select(dim, select_indices))

        results["scores"] = random_sampler.scores
        results["predictions"] = random_sampler.predictions
        results["attention"] = random_sampler.attention
        # yida translate
        # results["entropy"] = random_sampler.entropy
        # if self.model.pos_generator is not None:
        #     results["pos_predictions"] = random_sampler.pos_predictions
        return results

    def translate_batch(self, batch, src_vocabs, attn_debug,
                        # yida RL translate
                        memory_bank, src_lengths, enc_states, src, learned_t, sta=False, sample_method="freq"):
        """Translate a batch of sentences."""
        with torch.no_grad():
            return self._translate_random_sampling(
                batch,
                src_vocabs,
                self.max_length,
                # yida RL translate
                memory_bank, src_lengths, enc_states, src, learned_t, sta, sample_method,
                #
                min_length=self.min_length,
                sampling_temp=self.random_sampling_temp,
                keep_topk=self.sample_from_topk,
                return_attention=attn_debug or self.replace_unk)

    def _run_encoder(self, batch):
        src, src_lengths = batch.src if isinstance(batch.src, tuple) \
            else (batch.src, None)

        # yida translate
        if self.model.tag_enc:
            pos_src, _ = batch.tag_src if isinstance(batch.tag_src, tuple) else (batch.tag_src, None)
        else:
            pos_src = None
        enc_states, memory_bank, src_lengths = self.model.encoder(
            src, pos_src, src_lengths)
        if src_lengths is None:
            assert not isinstance(memory_bank, tuple), \
                'Ensemble decoding only supported for text data'
            src_lengths = torch.Tensor(batch.batch_size) \
                .type_as(memory_bank) \
                .long() \
                .fill_(memory_bank.size(0))
        return src, enc_states, memory_bank, src_lengths

    def _decode_and_generate(
            self,
            decoder_in,
            memory_bank,
            tag_src,
            tag_decoder_in,
            learned_t,
            batch,
            src_vocabs,
            memory_lengths,
            src_map=None,
            step=None,
            batch_offset=None):
        if self.copy_attn:
            # Turn any copied words into UNKs.
            decoder_in = decoder_in.masked_fill(
                decoder_in.gt(self._tgt_vocab_len - 1), self._tgt_unk_idx
            )

        # Decoder forward, takes [tgt_len, batch, nfeats] as input
        # and [src_len, batch, hidden] as memory_bank
        # in case of inference tgt_len = 1, batch = beam times batch_size
        # in case of Gold Scoring tgt_len = actual length, batch = 1 batch
        dec_out, dec_attn, rnn_outs = self.model.decoder(
            # yida translate
            decoder_in, memory_bank, self.model.generators.get("tag", None), tag_src, tag_decoder_in,
            memory_lengths=memory_lengths, step=step
        )

        # Generator forward.
        if "std" in dec_attn:
            attn = dec_attn["std"]
        else:
            attn = None

        tag_outputs = rnn_outs if not isinstance(rnn_outs, list) else dec_out
        if "tag" in self.model.generators:
            tag_log_probs = self.model.generators["tag"](tag_outputs.squeeze(0))
            tag_argmax = tag_log_probs.max(1)[1]
            # tag_logits = self.model.generators["tag"][0](tag_outputs.squeeze(0))
            # tag_log_probs = (tag_logits/0.5).log_softmax(dim=-1)
            # dist = torch.distributions.Multinomial(logits=tag_log_probs, total_count=1)
            # tag_argmax = torch.argmax(dist.sample(), dim=1)
        else:
            tag_log_probs = None
        pass_indices = {}
        if "high" in self.model.generators:
            log_probs = torch.full([dec_out.squeeze(0).shape[0], 50004], -float('inf'),
                                   dtype=torch.float, device=self._dev)
            for k, gen in self.model.generators.items():
                if k == "tag":
                    continue
                indices = tag_argmax.eq(self.tag_vocab[k])
                pass_indices[k] = indices
                if indices.any():
                    k_output = dec_out.squeeze(0)[indices]
                    k_logits = gen(k_output)
                    if learned_t is not None:
                        # k_logits = get_topp(k_logits)
                        k_logits /= learned_t[k][indices]
                    k_probs = torch.log_softmax(k_logits, dim=-1)
                    mask = indices.float().unsqueeze(-1).mm(self.tag_mask[k])
                    log_probs.masked_scatter_(mask.bool(), k_probs)
        else:
            logits = self.model.generators["generator"](dec_out.squeeze(0))
            high_num = self.tag_mask["high"].sum().long().item()
            high_indices = tag_argmax.eq(self.tag_vocab["high"])
            low_indices = tag_argmax.eq(self.tag_vocab["low"])
            pass_indices["high"] = high_indices
            pass_indices["low"] = low_indices
            logits[high_indices, high_num:] = -float("inf")
            logits[low_indices, :high_num] = -float("inf")
            if learned_t is not None:
                logits[high_indices] /= learned_t["high"][high_indices]
                logits[low_indices] /= learned_t["low"][low_indices]
                # logits[high_indices] /= 0.5174
                # logits[low_indices] /= 1.9034
            log_probs = torch.log_softmax(logits, dim=-1)
        # yida translate
        return log_probs, attn, tag_log_probs, pass_indices

    def _report_score(self, name, score_total, words_total):
        if words_total == 0:
            msg = "%s No words predicted" % (name,)
        else:
            msg = ("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
                name, score_total / words_total,
                name, math.exp(-score_total / words_total)))
        return msg

    def _report_bleu(self, tgt_path):
        import subprocess
        base_dir = os.path.abspath(__file__ + "/../../..")
        # Rollback pointer to the beginning.
        self.out_file.seek(0)
        print()

        res = subprocess.check_output(
            "perl %s/tools/multi-bleu.perl %s" % (base_dir, tgt_path),
            stdin=self.out_file, shell=True
        ).decode("utf-8")

        msg = ">> " + res.strip()
        return msg

    def _report_rouge(self, tgt_path):
        import subprocess
        path = os.path.split(os.path.realpath(__file__))[0]
        msg = subprocess.check_output(
            "python %s/tools/test_rouge.py -r %s -c STDIN" % (path, tgt_path),
            shell=True, stdin=self.out_file
        ).decode("utf-8").strip()
        return msg

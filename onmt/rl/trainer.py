#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function

import codecs
import math
import os
import random
import time
from itertools import count

import torch
from tensorboardX import SummaryWriter

import onmt.decoders.ensemble
import onmt.inputters as inputters
import onmt.model_builder
import onmt.rl.beam
from onmt.modules.copy_generator import collapse_copy_scores
from onmt.rl.beam_search import BeamSearch
# yida RL translate
from onmt.rl.eval import cal_reward
from onmt.rl.random_sampling import RandomSampling
from onmt.utils.misc import tile, set_random_seed


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
        # if "tgt" in batch.__dict__:
        if False:
            gs = self._score_target(
                batch, memory_bank, src_lengths, src_vocabs,
                batch.src_map if use_src_map else None)
            self.model.decoder.init_state(src, memory_bank, enc_states)
        else:
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
            data=[("src", test_src), ("pos_src", test_tag_src)]
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
            data=[("src", train_src), ("tgt", train_tgt), ("pos_src", train_tag_src), ("pos_tgt", train_tag_tgt)]
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
            data=[("src", valid_src), ("tgt", valid_tgt), ("pos_src", valid_tag_src), ("pos_tgt", valid_tag_tgt)]
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
        inputs = enc_states[-1].squeeze()
        logits_t = self.rl_model(inputs)
        low_logits_t = self.rl_model.generator(inputs) if self.rl_model.generator is not None else None

        # compute loss
        loss = self._compute_loss_k(logits_t, low_logits_t, batch, data, xlation_builder,
                                    src, enc_states, memory_bank, src_lengths)
        # loss.backward()
        if loss is not None:
            self.optim.backward(loss)
        self.optim.step()

    def _compute_loss_k(self, logits_t, low_logits_t, batch, data, xlation_builder, src, enc_states, memory_bank,
                        src_lengths):
        low_k_topk_ids = None
        if random.random() < (self.random_steps - self.optim.training_step) / self.random_steps:
            k_topk_ids = [torch.tensor([[random.randint(0, 19)] for i in range(batch.batch_size)],
                                       device=self._dev) for i in range(self.samples_n)]
            if low_logits_t is not None:
                low_k_topk_ids = [torch.tensor([[random.randint(0, 19)] for i in range(batch.batch_size)],
                                               device=self._dev) for i in range(self.samples_n)]
        else:
            dist = torch.distributions.Multinomial(logits=logits_t, total_count=1)
            k_topk_ids = [torch.argmax(dist.sample(), dim=1, keepdim=True) for i in range(self.samples_n)]
            if low_logits_t is not None:
                low_dist = torch.distributions.Multinomial(logits=low_logits_t, total_count=1)
                low_k_topk_ids = [torch.argmax(low_dist.sample(), dim=1, keepdim=True) for i in range(self.samples_n)]

        k_learned_t = self.tid2t(k_topk_ids)
        k_topk_ids = torch.stack(k_topk_ids, -1)
        # k_topk_ids = torch.cat(k_topk_ids, 0)
        k_logits_t = torch.stack([logits_t] * self.samples_n, -1)
        # k_logits_t = torch.cat([logits_t] * k, 0)
        loss_t = self.criterion(k_logits_t, k_topk_ids.squeeze()).mean(0)

        low_k_t = None
        if low_k_topk_ids is not None:
            low_k_t = self.tid2t(low_k_topk_ids)
            low_k_topk_ids = torch.stack(low_k_topk_ids, -1)
            # k_topk_ids = torch.cat(k_topk_ids, 0)
            low_k_logits_t = torch.stack([low_logits_t] * self.samples_n, -1)
            # k_logits_t = torch.cat([logits_t] * k, 0)
            loss_t += self.criterion(low_k_logits_t, low_k_topk_ids.squeeze()).mean(0)

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
                k_learned_t[i], low_k_t[i] if low_k_t else None, sample_method=self.samples_method
            )

            # translate, so slow
            batch_sents, golden_truth = self.ids2sents(batch_data, xlation_builder)

            # cal rewards
            reward_dict = cal_reward(batch_sents, golden_truth)
            if self.samples_method == "topk":
                k_reward_qs.append(reward_dict["bleu"])
            else:
                k_reward_qs.append(reward_dict["bleu"] + reward_dict["dist"] / 100)
            #
            k_bleu.append(reward_dict["bleu"])
            k_dist.append(reward_dict["dist"])

        # with torch.no_grad():
        #     self.model.decoder.init_state(src, memory_bank, enc_states)
        # batch_bl_data = self.translate_batch(
        #     batch, data.src_vocabs, attn_debug, memory_bank, src_lengths, enc_states, src, k_learned_t[0], bl=True
        # )
        # baseline, _ = self.ids2sents(batch_bl_data, xlation_builder)

        # topk_scores, topk_ids = logits_t.topk(1, dim=-1)
        # bl_t = self.tid2t([topk_ids])
        # with torch.no_grad():
        #     self.model.decoder.init_state(src, memory_bank, enc_states)
        # bl_batch_data = self.translate_batch(
        #     batch, data.src_vocabs, attn_debug, memory_bank, src_lengths, enc_states, src,
        #     bl_t[0], sample_method=self.samples_method
        # )
        # baseline, golden_truth = self.ids2sents(bl_batch_data, xlation_builder)
        #
        # reward_bl_dict = cal_reward(baseline, golden_truth)
        # if self.samples_method == "topk":
        #     reward_bl = reward_bl_dict["bleu"]
        # else:
        #     reward_bl = reward_bl_dict["bleu"] + reward_bl_dict["dist"] / 100
        # reward_bl = reward_bl_dict["bleu"]

        reward_mean = sum(k_reward_qs) / len(k_reward_qs)
        bleu_mean = sum(k_bleu) / len(k_bleu)
        dist_mean = sum(k_dist) / len(k_dist)

        reward_bl = reward_mean
        reward = (torch.tensor(k_reward_qs, device=self._dev) - reward_bl) / \
                 max([abs(x - reward_bl) for x in k_reward_qs])
        loss = reward * loss_t

        self.writer.add_scalars("train_t/t", {"t_mean": k_learned_t.mean(),
                                              "mean+stc": k_learned_t.mean() + k_learned_t.std(),
                                              "mean-stc": k_learned_t.mean() - k_learned_t.std()},
                                self.optim.training_step)
        if low_k_t is not None:
            self.writer.add_scalars("train_t/low_t", {"t_mean": low_k_t.mean(),
                                                      "mean+stc": low_k_t.mean() + low_k_t.std(),
                                                      "mean-stc": low_k_t.mean() - low_k_t.std()},
                                    self.optim.training_step)
        self.writer.add_scalars("train_loss", {"loss_mean": loss_t.mean()}, self.optim.training_step)
        self.writer.add_scalars("train_reward/reward_mean", {"reward_mean": reward_mean}, self.optim.training_step)
        self.writer.add_scalars("train_reward/bleu_mean", {"bleu_mean": bleu_mean}, self.optim.training_step)
        self.writer.add_scalars("train_reward/dist_mean", {"dist_mean": dist_mean}, self.optim.training_step)
        # self.writer.add_scalars("train_reward/reward_bl", {"reward_bl": reward_bl}, self.optim.training_step)
        # self.writer.add_scalars("train_reward/bleu_bl", {"bleu_bl": reward_bl_dict["bleu"]}, self.optim.training_step)
        self.writer.add_scalars("lr", {"lr": self.optim.learning_rate()}, self.optim.training_step)
        return loss

    def tid2t(self, t_ids):
        # return t_ids.float() + 0.001
        t = (torch.stack(t_ids, 0).float() + 1) / 10
        return t

    def validate(self, valid_iter, data, xlation_builder, infer=False, attn_debug=False):
        loss_total = 0.0
        step = 0
        all_predictions = []
        arg_prediction = []
        golden = []

        self.rl_model.eval()
        with torch.no_grad():
            for batch in valid_iter:
                src, enc_states, memory_bank, src_lengths = self._run_encoder(batch)
                self.model.decoder.init_state(src, memory_bank, enc_states)
                # input = enc_states.transpose(0, 1)
                # input = input.contiguous().view(input.size(0), -1)
                inputs = enc_states[-1].squeeze()

                # F-prop through the model.
                logits_t = self.rl_model(inputs)
                low_logits_t = self.rl_model.generator(inputs) if self.rl_model.generator is not None else None
                if not (infer or self.sta):
                    dist = torch.distributions.Multinomial(logits=logits_t, total_count=1)
                    k_topk_ids = torch.argmax(dist.sample(), dim=1, keepdim=True)
                    low_t = None
                    if low_logits_t is not None:
                        low_dist = torch.distributions.Multinomial(logits=low_logits_t, total_count=1)
                        low_k_topk_ids = torch.argmax(low_dist.sample(), dim=1, keepdim=True)
                        low_t = self.tid2t([low_k_topk_ids])
                    learned_t = self.tid2t([k_topk_ids])
                    batch_data = self.translate_batch(
                        batch, data.src_vocabs, attn_debug, memory_bank, src_lengths, enc_states, src,
                        learned_t[0], low_t[0] if low_t else None, sample_method=self.samples_method
                    )
                    batch_sents, golden_truth = self.ids2sents(batch_data, xlation_builder)
                    all_predictions += batch_sents
                    golden += golden_truth

                    loss_t = self.criterion(logits_t, k_topk_ids.view(-1))
                    loss = loss_t.mean()
                    loss_total += loss

                # arg
                topk_scores, topk_ids = logits_t.topk(1, dim=-1)
                learned_t_arg = self.tid2t([topk_ids])
                low_t_arg = None
                if low_logits_t is not None:
                    low_topk_scores, low_topk_ids = low_logits_t.topk(1, dim=-1)
                    low_t_arg = self.tid2t([low_topk_ids])
                self.model.decoder.init_state(src, memory_bank, enc_states)
                batch_data_arg = self.translate_batch(
                    batch, data.src_vocabs, attn_debug, memory_bank, src_lengths, enc_states, src,
                    learned_t_arg[0], low_t_arg[0] if low_t_arg is not None else None,
                    sample_method=self.samples_method, sta=self.sta)
                # translate, so slow
                if not self.sta:
                    batch_sents_arg, _ = self.ids2sents(batch_data_arg, xlation_builder, infer)
                    arg_prediction += batch_sents_arg

                step += 1
        if infer:
            return arg_prediction
        reward_qs_dict_arg = cal_reward(arg_prediction, golden)
        reward_qs_dict = cal_reward(all_predictions, golden)
        if self.samples_method == "freq":
            reward = (reward_qs_dict["bleu"] + reward_qs_dict["dist"] / 100)
            reward_arg = (reward_qs_dict_arg["bleu"] + reward_qs_dict_arg["dist"] / 100)
        else:
            reward = reward_qs_dict["bleu"]
            reward_arg = reward_qs_dict_arg["bleu"]
        self.writer.add_scalars("valid_loss", {"loss": loss_total / step}, self.optim.training_step)
        self.writer.add_scalars("valid_reward/reward",
                                {"reward_sample": reward, "reward_arg": reward_arg},
                                self.optim.training_step)
        self.writer.add_scalars("valid_reward/bleu",
                                {"bleu": reward_qs_dict["bleu"],
                                 "bleu_arg": reward_qs_dict_arg["bleu"]}, self.optim.training_step)
        self.writer.add_scalars("valid_reward/dist",
                                {"dist": reward_qs_dict["dist"],
                                 "dist_arg": reward_qs_dict_arg["dist"]}, self.optim.training_step)
        self.writer.add_scalars("valid_t/sample",
                                {"mean": learned_t[0].mean(),
                                 "mean+std": learned_t[0].mean() + learned_t[0].std(),
                                 "mean-std": learned_t[0].mean() - learned_t[0].std()},
                                self.optim.training_step)

        self.writer.add_scalars("valid_t/arg",
                                {"mean": learned_t_arg[0].mean(),
                                 "mean+std": learned_t_arg[0].mean() + learned_t_arg[0].std(),
                                 "mean-std": learned_t_arg[0].mean() - learned_t_arg[0].std()},
                                self.optim.training_step)
        if low_t is not None:
            self.writer.add_scalars("valid_t/low_sample",
                                    {"mean": low_t[0].mean(),
                                     "mean+std": low_t[0].mean() + low_t[0].std(),
                                     "mean-std": low_t[0].mean() - low_t[0].std()},
                                    self.optim.training_step)
            self.writer.add_scalars("valid_t/low_arg",
                                    {"mean": low_t_arg[0].mean(),
                                     "mean+std": low_t_arg[0].mean() + low_t_arg[0].std(),
                                     "mean-std": low_t_arg[0].mean() - low_t_arg[0].std()},
                                    self.optim.training_step)
            print("valid_low_t:", low_t_arg[0].mean(), "std:", low_t_arg[0].std(), self.optim.training_step)
        print("valid_t:", learned_t_arg[0].mean(), "std:", learned_t_arg[0].std(), self.optim.training_step)
        print("valid loss:", loss_total / step)

        self.rl_model.train()

    def ids2sents(
            self,
            batch_data,
            xlation_builder,
            infer=False):
        """

        :param batch_data:
        :param xlation_builder:
        :param all_predictions:
        :param all_entropy:
        :param all_pos_predictions:
        :param all_pos_entropy:
        :param cnt_high:
        :param cnt_line:
        :return:
        """
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
            memory_bank, src_lengths, enc_states, src, learned_t, low_t, sta, sample_method,
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

        tag_src, _ = batch.pos_src if isinstance(batch.pos_src, tuple) else (batch.pos_src, None)

        random_sampler = RandomSampling(
            self._tgt_pad_idx, self._tgt_bos_idx, self._tgt_eos_idx,
            batch_size, mb_device, min_length, self.block_ngram_repeat,
            self._exclusion_idxs, return_attention, self.max_length,
            sampling_temp, keep_topk, memory_lengths,
            # yida translate
            self.model.tag_generator is not None, vocab_pos, learned_t, low_t, sample_method, tag_src)

        for step in range(max_length):
            # Shape: (1, B, 1)
            decoder_input = random_sampler.alive_seq[:, -1].view(1, -1, 1)
            tag_decoder_src = random_sampler.tag_alive_src \
                if self.model.low_generator is not None else None
            tag_decoder_in = random_sampler.pos_alive_seq[:, -1].view(1, -1, 1) \
                if self.model.tag_generator is not None else None
            t_in = random_sampler.learned_t
            low_t_in = random_sampler.low_t

            # yida translate
            log_probs, attn, pos_log_probs = self._decode_and_generate(
                decoder_input,
                memory_bank,
                # yida tranlate
                tag_decoder_src,
                tag_decoder_in,
                t_in,
                low_t_in,
                batch,
                src_vocabs,
                memory_lengths=memory_lengths,
                src_map=src_map,
                step=step,
                batch_offset=random_sampler.select_indices
            )

            # yida tranlate
            random_sampler.advance(log_probs, attn, pos_log_probs, sta)
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
                        memory_bank, src_lengths, enc_states, src, learned_t, low_t, sta=False, sample_method="freq"):
        """Translate a batch of sentences."""
        with torch.no_grad():
            return self._translate_random_sampling(
                batch,
                src_vocabs,
                self.max_length,
                # yida RL translate
                memory_bank, src_lengths, enc_states, src, learned_t, low_t, sta, sample_method,
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
            pos_src, _ = batch.pos_src if isinstance(batch.pos_src, tuple) else (batch.pos_src, None)
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
            low_t,
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
            decoder_in, memory_bank, self.model.tag_generator, tag_src, tag_decoder_in,
            memory_lengths=memory_lengths, step=step
        )

        # Generator forward.
        if not self.copy_attn:
            if "std" in dec_attn:
                attn = dec_attn["std"]
            else:
                attn = None

            tag_outputs = rnn_outs if not isinstance(rnn_outs, list) else dec_out
            tag_log_probs = self.model.tag_generator(
                tag_outputs.squeeze(0)) if self.model.tag_generator is not None else None
            tag_argmax = tag_log_probs.max(1)[1]
            if self.model.low_generator is not None:
                high_indices = tag_argmax.eq(4)
                low_indices = tag_argmax.eq(5)
                high_out = dec_out.squeeze(0)[high_indices]
                low_out = dec_out.squeeze(0)[low_indices]
                assert high_out.shape[0] + low_out.shape[0] == tag_argmax.shape[0]
                high_num = self.model.generator._modules["0"].out_features
                low_num = self.model.low_generator._modules["0"].out_features
                log_probs = torch.full([dec_out.squeeze(0).shape[0], high_num + low_num], -float('inf'),
                                       dtype=torch.float, device=high_out.device)
                if high_out.shape[0] > 0:
                    high_logits = self.model.generator(high_out)
                    high_probs = torch.log_softmax(high_logits / learned_t[high_indices], dim=-1)
                    log_probs[high_indices, :high_num] = high_probs
                if low_out.shape[0] > 0:
                    low_logits = self.model.low_generator(low_out)
                    low_logits /= low_t[low_indices] if low_t is not None else learned_t[low_indices]
                    low_probs = torch.log_softmax(low_logits, dim=-1)
                    log_probs[low_indices, high_num:] = low_probs
            else:
                logits = self.model.generator(dec_out.squeeze(0))
                # log_probs = torch.log_softmax(logits, dim=-1)
                log_probs = logits
            # log_probs = self.model.generator(dec_out.squeeze(0))
            # # yida translate
            # pos_log_probs = self.model.tag_generator(
            #     dec_out.squeeze(0)) if self.model.tag_generator is not None else None
            # returns [(batch_size x beam_size) , vocab ] when 1 step
            # or [ tgt_len, batch_size, vocab ] when full sentence
        else:
            attn = dec_attn["copy"]
            scores = self.model.generator(dec_out.view(-1, dec_out.size(2)),
                                          attn.view(-1, attn.size(2)),
                                          src_map)
            # here we have scores [tgt_lenxbatch, vocab] or [beamxbatch, vocab]
            if batch_offset is None:
                scores = scores.view(-1, batch.batch_size, scores.size(-1))
                scores = scores.transpose(0, 1).contiguous()
            else:
                scores = scores.view(-1, self.beam_size, scores.size(-1))
            scores = collapse_copy_scores(
                scores,
                batch,
                self._tgt_vocab,
                src_vocabs,
                batch_dim=0,
                batch_offset=batch_offset
            )
            scores = scores.view(decoder_in.size(0), -1, scores.size(-1))
            log_probs = scores.squeeze(0).log()
            # returns [(batch_size x beam_size) , vocab ] when 1 step
            # or [ tgt_len, batch_size, vocab ] when full sentence
        # yida translate
        return log_probs, attn, tag_log_probs

    def _translate_batch(
            self,
            batch,
            src_vocabs,
            max_length,
            min_length=0,
            ratio=0.,
            n_best=1,
            return_attention=False):
        # TODO: support these blacklisted features.
        assert not self.dump_beam

        # (0) Prep the components of the search.
        use_src_map = self.copy_attn
        beam_size = self.beam_size
        batch_size = batch.batch_size

        # (1) Run the encoder on the src.
        src, enc_states, memory_bank, src_lengths = self._run_encoder(batch)
        self.model.decoder.init_state(src, memory_bank, enc_states)

        results = {
            "predictions": None,
            "scores": None,
            "attention": None,
            "batch": batch,
            "gold_score": self._gold_score(
                batch, memory_bank, src_lengths, src_vocabs, use_src_map,
                enc_states, batch_size, src)}

        # (2) Repeat src objects `beam_size` times.
        # We use batch_size x beam_size
        src_map = (tile(batch.src_map, beam_size, dim=1)
                   if use_src_map else None)
        self.model.decoder.map_state(
            lambda state, dim: tile(state, beam_size, dim=dim))

        if isinstance(memory_bank, tuple):
            memory_bank = tuple(tile(x, beam_size, dim=1) for x in memory_bank)
            mb_device = memory_bank[0].device
        else:
            memory_bank = tile(memory_bank, beam_size, dim=1)
            mb_device = memory_bank.device
        memory_lengths = tile(src_lengths, beam_size)

        # (0) pt 2, prep the beam object
        beam = BeamSearch(
            beam_size,
            n_best=n_best,
            batch_size=batch_size,
            global_scorer=self.global_scorer,
            pad=self._tgt_pad_idx,
            eos=self._tgt_eos_idx,
            bos=self._tgt_bos_idx,
            min_length=min_length,
            ratio=ratio,
            max_length=max_length,
            mb_device=mb_device,
            return_attention=return_attention,
            stepwise_penalty=self.stepwise_penalty,
            block_ngram_repeat=self.block_ngram_repeat,
            exclusion_tokens=self._exclusion_idxs,
            memory_lengths=memory_lengths)

        for step in range(max_length):
            decoder_input = beam.current_predictions.view(1, -1, 1)

            log_probs, attn = self._decode_and_generate(
                decoder_input,
                memory_bank,
                batch,
                src_vocabs,
                memory_lengths=memory_lengths,
                src_map=src_map,
                step=step,
                batch_offset=beam._batch_offset)

            beam.advance(log_probs, attn)
            any_beam_is_finished = beam.is_finished.any()
            if any_beam_is_finished:
                beam.update_finished()
                if beam.done:
                    break

            select_indices = beam.current_origin

            if any_beam_is_finished:
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

        results["scores"] = beam.scores
        results["predictions"] = beam.predictions
        results["attention"] = beam.attention
        return results

    # This is left in the code for now, but unsued
    def _translate_batch_deprecated(self, batch, src_vocabs):
        # (0) Prep each of the components of the search.
        # And helper method for reducing verbosity.
        use_src_map = self.copy_attn
        beam_size = self.beam_size
        batch_size = batch.batch_size

        beam = [onmt.translate.Beam(
            beam_size,
            n_best=self.n_best,
            cuda=self.cuda,
            global_scorer=self.global_scorer,
            pad=self._tgt_pad_idx,
            eos=self._tgt_eos_idx,
            bos=self._tgt_bos_idx,
            min_length=self.min_length,
            stepwise_penalty=self.stepwise_penalty,
            block_ngram_repeat=self.block_ngram_repeat,
            exclusion_tokens=self._exclusion_idxs)
            for __ in range(batch_size)]

        # (1) Run the encoder on the src.
        src, enc_states, memory_bank, src_lengths = self._run_encoder(batch)
        self.model.decoder.init_state(src, memory_bank, enc_states)

        results = {
            "predictions": [],
            "scores": [],
            "attention": [],
            "batch": batch,
            "gold_score": self._gold_score(
                batch, memory_bank, src_lengths, src_vocabs, use_src_map,
                enc_states, batch_size, src)}

        # (2) Repeat src objects `beam_size` times.
        # We use now  batch_size x beam_size (same as fast mode)
        src_map = (tile(batch.src_map, beam_size, dim=1)
                   if use_src_map else None)
        self.model.decoder.map_state(
            lambda state, dim: tile(state, beam_size, dim=dim))

        if isinstance(memory_bank, tuple):
            memory_bank = tuple(tile(x, beam_size, dim=1) for x in memory_bank)
        else:
            memory_bank = tile(memory_bank, beam_size, dim=1)
        memory_lengths = tile(src_lengths, beam_size)

        # (3) run the decoder to generate sentences, using beam search.
        for i in range(self.max_length):
            if all((b.done for b in beam)):
                break

            # (a) Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.

            inp = torch.stack([b.current_predictions for b in beam])
            inp = inp.view(1, -1, 1)

            # (b) Decode and forward
            out, beam_attn = self._decode_and_generate(
                inp, memory_bank, batch, src_vocabs,
                memory_lengths=memory_lengths, src_map=src_map, step=i
            )
            out = out.view(batch_size, beam_size, -1)
            beam_attn = beam_attn.view(batch_size, beam_size, -1)

            # (c) Advance each beam.
            select_indices_array = []
            # Loop over the batch_size number of beam
            for j, b in enumerate(beam):
                if not b.done:
                    b.advance(out[j, :],
                              beam_attn.data[j, :, :memory_lengths[j]])
                select_indices_array.append(
                    b.current_origin + j * beam_size)
            select_indices = torch.cat(select_indices_array)

            self.model.decoder.map_state(
                lambda state, dim: state.index_select(dim, select_indices))

        # (4) Extract sentences from beam.
        for b in beam:
            scores, ks = b.sort_finished(minimum=self.n_best)
            hyps, attn = [], []
            for times, k in ks[:self.n_best]:
                hyp, att = b.get_hyp(times, k)
                hyps.append(hyp)
                attn.append(att)
            results["predictions"].append(hyps)
            results["scores"].append(scores)
            results["attention"].append(attn)

        return results

    def _score_target(self, batch, memory_bank, src_lengths,
                      src_vocabs, src_map):
        tgt = batch.tgt
        tgt_in = tgt[:-1]

        log_probs, attn = self._decode_and_generate(
            tgt_in, memory_bank, batch, src_vocabs,
            memory_lengths=src_lengths, src_map=src_map)

        log_probs[:, :, self._tgt_pad_idx] = 0
        gold = tgt[1:]
        gold_scores = log_probs.gather(2, gold)
        gold_scores = gold_scores.sum(dim=0).view(-1)

        return gold_scores

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

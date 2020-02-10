#!/usr/bin/env python
"""Training on a single process."""
import os
from itertools import repeat

import torch

from onmt.inputters.inputter import load_old_vocab, old_style_vocab
from onmt.rl.model_builder import build_model
from onmt.rl.model_saver import build_model_saver
from onmt.rl.step_trainer import build_rltor_dec
from onmt.rl.trainer import build_rltor_enc
from onmt.utils.logging import logger
from onmt.utils.misc import set_random_seed
from onmt.utils.misc import split_corpus
from onmt.utils.optimizers import Optimizer
from onmt.utils.parse import ArgumentParser


def _check_save_model_path(opt):
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)


def _tally_parameters(model):
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        else:
            dec += param.nelement()
    return enc + dec, enc, dec


def configure_process(opt, device_id):
    if device_id >= 0:
        torch.cuda.set_device(device_id)
    set_random_seed(opt.seed, device_id >= 0)


def main(opt, device_id, batch_queue=None, semaphore=None):
    # NOTE: It's important that ``opt`` has been validated and updated
    # at this point.
    # configure_process(opt, device_id)
    # init_logger(opt.log_file)
    # assert len(opt.accum_count) == len(opt.accum_steps), \
    #     'Number of accum_count values must match number of accum_steps'
    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        logger.info('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)
        model_opt = ArgumentParser.ckpt_model_opts(checkpoint["opt"])
        ArgumentParser.update_model_opts(model_opt)
        # ArgumentParser.validate_model_opts(model_opt)
        logger.info('Loading vocab from checkpoint at %s.' % opt.train_from)
        vocab = checkpoint['vocab']
    else:
        checkpoint = None
        model_opt = opt
        vocab = torch.load(opt.data + '.vocab.pt')

    # check for code where vocab is saved instead of fields
    # (in the future this will be done in a smarter way)
    if old_style_vocab(vocab):
        fields = load_old_vocab(
            vocab, opt.model_type, dynamic_dict=opt.copy_attn)
    else:
        fields = vocab

    # Report src and tgt vocab sizes, including for features
    for side in ['src', 'tgt']:
        f = fields[side]
        try:
            f_iter = iter(f)
        except TypeError:
            f_iter = [(side, f)]
        for sn, sf in f_iter:
            if sf.use_vocab:
                logger.info(' * %s vocab size = %d' % (sn, len(sf.vocab)))

    # Build model.
    rl_model = build_model(model_opt, opt, fields, checkpoint)

    _check_save_model_path(opt)

    # Build optimizer.
    # optim = torch.optim.Adam(rl_model.parameters())
    optim = Optimizer.from_opt(rl_model, opt, checkpoint=checkpoint)

    # Build model saver
    model_saver = build_model_saver(model_opt, opt, rl_model, optim)
    # model_saver = None

    # trainer = build_trainer(
    #     opt, device_id, model, fields, optim, model_saver=model_saver)
    build_rltor = build_rltor_enc if not opt.rl_step else build_rltor_dec
    rltor = build_rltor(opt, rl_model, optim, model_saver, report_score=False)

    src_shards = split_corpus(opt.src, opt.shard_size)
    tgt_shards = split_corpus(opt.tgt, opt.shard_size) \
        if opt.tgt is not None else repeat(None)
    if opt.infer:
        for i, src_shard in enumerate(src_shards):
            logger.info("Translating shard %d." % i)
            rltor.infer(
                src_shard,
                batch_size=opt.batch_size,
                batch_type=opt.batch_type)
    else:
        valid_src_shards = split_corpus(opt.valid_src, opt.shard_size)
        valid_tgt_shards = split_corpus(opt.valid_tgt, opt.shard_size)

        tag_src_shards = split_corpus(opt.tag_src, opt.shard_size) \
            if opt.tag_tgt is not None else repeat(None)
        valid_tag_src_shards = split_corpus(opt.valid_tag_src, opt.shard_size) \
            if opt.valid_tag_tgt is not None else repeat(None)
        tag_tgt_shards = split_corpus(opt.tag_tgt, opt.shard_size) \
            if opt.tag_tgt is not None else repeat(None)
        valid_tag_tgt_shards = split_corpus(opt.valid_tag_tgt, opt.shard_size) \
            if opt.valid_tag_tgt is not None else repeat(None)

        shard_pairs = zip(src_shards, tgt_shards, tag_src_shards, tag_tgt_shards,
                          valid_src_shards, valid_tgt_shards, valid_tag_src_shards,
                          valid_tag_tgt_shards)

        for i, (train_src_shard, train_tgt_shard, train_tag_src_shard, train_tag_tgt_shard,
                valid_src_shard, valid_tgt_shard, valid_tag_src_shard, valid_tag_tgt_shard) in enumerate(shard_pairs):
            logger.info("Learning shard %d." % i)
            rltor.train(
                train_src_shard,
                train_tgt_shard,
                train_tag_src_shard,
                train_tag_tgt_shard,
                valid_src_shard,
                valid_tgt_shard,
                valid_tag_src_shard,
                valid_tag_tgt_shard,
                src_dir=opt.src_dir,
                batch_size=opt.batch_size,
                batch_type=opt.batch_type
            )

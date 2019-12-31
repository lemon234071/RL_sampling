#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import onmt.opts as opts
from onmt.rl.rl_train_single import main as single_main
from onmt.utils.logging import init_logger
from onmt.utils.parse import ArgumentParser


def rl_train(opt):
    # ArgumentParser.validate_train_opts(opt)
    ArgumentParser.validate_translate_opts(opt)
    logger = init_logger(opt.log_file)

    single_main(opt, 0)


def _get_parser():
    parser = ArgumentParser(description='RL_train.py')

    opts.config_opts(parser)
    # yida RL
    opts.model_opts(parser)
    # opts.train_opts(parser)
    opts.translate_opts(parser)
    return parser


def main():
    parser = _get_parser()

    opt = parser.parse_args()
    rl_train(opt)


if __name__ == "__main__":
    main()

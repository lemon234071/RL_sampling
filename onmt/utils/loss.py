"""
This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

import onmt
from utils import *


def build_loss_compute(model, fields, opt, train=True):
    """
    Returns a LossCompute subclass which wraps around an nn.Module subclass
    (such as nn.NLLLoss) which defines the loss criterion. The LossCompute
    object allows this loss to be computed in shards and passes the relevant
    data to a Statistics object which handles training/validation logging.
    Currently, the NMTLossCompute class handles all loss computation except
    for when using a copy mechanism.
    """
    device = torch.device("cuda" if onmt.utils.misc.use_gpu(opt) else "cpu")

    tgt_field = dict(fields)["tgt"].base_field
    padding_idx = tgt_field.vocab.stoi[tgt_field.pad_token]
    unk_idx = tgt_field.vocab.stoi[tgt_field.unk_token]

    if opt.lambda_coverage != 0:
        assert opt.coverage_attn, "--coverage_attn needs to be set in " \
                                  "order to use --lambda_coverage != 0"

    # if opt.copy_attn:
    #     criterion = onmt.modules.CopyGeneratorLoss(
    #         len(tgt_field.vocab), opt.copy_attn_force,
    #         unk_index=unk_idx, ignore_index=padding_idx
    #     )
    # elif opt.label_smoothing > 0 and train:
    #     criterion = LabelSmoothingLoss(
    #         opt.label_smoothing, len(tgt_field.vocab), ignore_index=padding_idx
    #     )
    # elif isinstance(model.generator[-1], LogSparsemax):
    #     criterion = SparsemaxLoss(ignore_index=padding_idx, reduction='sum')
    # else:
    #     criterion = nn.NLLLoss(ignore_index=padding_idx, reduction='sum')

    criterion = nn.NLLLoss(ignore_index=padding_idx, reduction='sum')
    criterions = {}
    ls_gen = opt.ls_gen.split(",")
    for k in model.generators.keys():
        if k in ls_gen and opt.label_smoothing > 0 and train:
            criterions[k] = LabelSmoothingLoss(
                opt.label_smoothing, model.generators[k]._modules["0"].out_features, ignore_index=padding_idx
            )
        else:
            criterions[k] = criterion
        criterions[k].to(device)

    # if the loss function operates on vectors of raw logits instead of
    # probabilities, only the first part of the generator needs to be
    # passed to the NMTLossCompute. At the moment, the only supported
    # loss function of this kind is the sparsemax loss.

    # use_raw_logits = isinstance(criterion, SparsemaxLoss)
    # loss_gen = model.generator[0] if use_raw_logits else model.generator
    loss_gen = None

    tag_field = dict(fields)["tag_tgt"].base_field
    compute = NMTLossCompute(criterion, loss_gen,
                             criterions, model.generators, opt.itoj, opt.statistic, tag_field.vocab.stoi, opt.high_num,
                             device, train=train,
                             lambda_coverage=opt.lambda_coverage)
    # if opt.copy_attn:
    #     compute = onmt.modules.CopyGeneratorLossCompute(
    #         criterion, loss_gen, tgt_field.vocab, opt.copy_loss_by_seqlength,
    #         lambda_coverage=opt.lambda_coverage
    #     )
    # else:
    #     compute = NMTLossCompute(criterion, loss_gen, model.generators, opt.itoj,
    #                              opt.statistic, tag_field.vocab.stoi, device, train=train,
    #                              lambda_coverage=opt.lambda_coverage)
    compute.to(device)

    return compute


class LossComputeBase(nn.Module):
    """
    Class for managing efficient loss computation. Handles
    sharding next step predictions and accumulating multiple
    loss computations

    Users can implement their own loss computation strategy by making
    subclass of this one.  Users need to implement the _compute_loss()
    and make_shard_state() methods.

    Args:
        generator (:obj:`nn.Module`) :
             module that maps the output of the decoder to a
             distribution over the target vocabulary.
        tgt_vocab (:obj:`Vocab`) :
             torchtext vocab object representing the target output
        normalzation (str): normalize by "sents" or "tokens"
    """

    # TODO(yida) loss
    def __init__(self, criterion, generator):
        super(LossComputeBase, self).__init__()
        self.criterion = criterion
        self.generator = generator

    @property
    def padding_idx(self):
        return self.criterion.ignore_index

    def _make_shard_state(self, batch, output, range_, attns=None):
        """
        Make shard state dictionary for shards() to return iterable
        shards for efficient loss computation. Subclass must define
        this method to match its own _compute_loss() interface.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            range_: the range of examples for computing, the whole
                    batch or a trunc of it?
            attns: the attns dictionary returned from the model.
        """
        return NotImplementedError

    def _compute_loss(self, batch, output, target, **kwargs):
        """
        Compute the loss. Subclass must define this method.

        Args:

            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        """
        return NotImplementedError

    def __call__(self,
                 batch,
                 output,
                 attns,
                 rnn_outs,
                 normalization=1.0,
                 shard_size=0,
                 trunc_start=0,
                 trunc_size=None):
        """Compute the forward loss, possibly in shards in which case this
        method also runs the backward pass and returns ``None`` as the loss
        value.

        Also supports truncated BPTT for long sequences by taking a
        range in the decoder output sequence to back propagate in.
        Range is from `(trunc_start, trunc_start + trunc_size)`.

        Note sharding is an exact efficiency trick to relieve memory
        required for the generation buffers. Truncation is an
        approximate efficiency trick to relieve the memory required
        in the RNN buffers.

        Args:
          batch (batch) : batch of labeled examples
          output (:obj:`FloatTensor`) :
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict) : dictionary of attention distributions
              `[tgt_len x batch x src_len]`
          normalization: Optional normalization factor.
          shard_size (int) : maximum number of examples in a shard
          trunc_start (int) : starting position of truncation window
          trunc_size (int) : length of truncation window

        Returns:
            A tuple with the loss and a :obj:`onmt.utils.Statistics` instance.
        """
        if trunc_size is None:
            trunc_size = batch.tgt.size(0) - trunc_start
        trunc_range = (trunc_start, trunc_start + trunc_size)
        shard_state = self._make_shard_state(batch, output, trunc_range, attns, rnn_outs=rnn_outs)
        if shard_size == 0:
            loss, stats = self._compute_loss(batch, **shard_state)
            return loss / float(normalization), stats
        batch_stats = onmt.utils.Statistics()
        for shard in shards(shard_state, shard_size):
            loss, stats = self._compute_loss(batch, **shard)
            with torch.enable_grad():
                loss.div(float(normalization)).backward()
            batch_stats.update(stats)
        return None, batch_stats

    def _stats(self, loss, scores, target):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.
        """
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target).masked_select(non_padding).sum().item()
        num_non_padding = non_padding.sum().item()
        # return onmt.utils.Statistics(loss, num_non_padding, num_correct)
        # TODO(yida) loss
        return onmt.utils.Statistics(loss["loss"].clone().item(),
                                     loss["tag_loss"].clone().item(),
                                     num_non_padding, num_correct)

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='sum')


class NMTLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """

    # TODO(yida) loss
    def __init__(self, criterion, generator,
                 criterions, generators, itoj_path, sta, tag_vocab, high_num,
                 device,
                 train=False,
                 normalization="sents",
                 lambda_coverage=0.0):
        super(NMTLossCompute, self).__init__(criterion, generator)
        self.lambda_coverage = lambda_coverage
        self.generator = generator

        self.criterions = criterions
        self.generators = generators
        self.itoj = load_json(itoj_path) if itoj_path != "" else None
        self.high_num = high_num
        self.tag_vocab = tag_vocab
        self.sta = sta
        self.writer = SummaryWriter(comment="sta_train") if train else SummaryWriter(comment="sta_valid")
        self.step = 0
        self.device = device
        self.train = train

    def _make_shard_state(self, batch, output, range_, attns=None, rnn_outs=None):
        # TODO(yida) loss
        if "tag" in self.generators.keys():
            shard_state = {
                "output": output,
                "target": batch.tgt[range_[0] + 1: range_[1], :, 0],
                "tag_output": rnn_outs if not isinstance(rnn_outs, list) else output.clone(),  # TODO clone or not
                "tag_target": batch.tag_tgt[range_[0] + 1: range_[1], :, 0]
            }
        else:
            shard_state = {
                "output": output,
                "target": batch.tgt[range_[0] + 1: range_[1], :, 0],
            }
        if self.lambda_coverage != 0.0:
            coverage = attns.get("coverage", None)
            std = attns.get("std", None)
            assert attns is not None
            assert std is not None, "lambda_coverage != 0.0 requires " \
                                    "attention mechanism"
            assert coverage is not None, "lambda_coverage != 0.0 requires " \
                                         "coverage attention"

            shard_state.update({
                "std_attn": attns.get("std"),
                "coverage_attn": coverage
            })
        return shard_state

    def _compute_loss(self, batch, output, target,
                      rnn_out=None, tag_output=None, tag_target=None,
                      std_attn=None, coverage_attn=None):
        # TODO(yida) temp rl
        # with torch.enable_grad():
        loss_dict = {"loss": torch.tensor(0.0).to(self.device),
                     "tag_loss": torch.tensor(0.0).to(self.device)}
        bottled_output = self._bottle(output)
        gtruth = target.view(-1)
        tag_scores = None

        # log_probs = torch.full([dec_out.squeeze(0).shape[0], high_num + low_num], -float('inf'),
        #                        dtype=torch.float, device=high_out.device)
        # log_probs[high_indices, :high_probs.shape[1]] = high_probs
        if "tag" in self.generators.keys():
            tag_bottled_output = self._bottle(tag_output)
            tag_scores = self.generators["tag"](tag_bottled_output)
            tag_gtruth = tag_target.view(-1)
            loss_dict["tag_loss"] = self.criterions["tag"](tag_scores, tag_gtruth)
            # sta
            tag_pred_indices = tag_scores.max(1)[1]
            tag_non_padding = tag_gtruth.ne(self.padding_idx)
            num_correct_tag = tag_pred_indices.eq(tag_gtruth).masked_select(tag_non_padding).sum().item()
            num_non_padding_tag = tag_non_padding.sum().item()
            self.writer.add_scalars("sta_acc",
                                    {"acc": 100 * num_correct_tag / num_non_padding_tag},
                                    self.step)
            for k, gen in self.generators.items():
                if k == "tag":
                    continue
                indices = tag_gtruth.eq(self.tag_vocab[k])
                if indices.any():
                    k_output = bottled_output[indices]
                    k_gtruth = gtruth[indices]
                    # if k == "0":
                    #     k_gtruth = k_gtruth - 154
                    k_gtruth = torch.tensor([self.itoj[i] for i in k_gtruth], dtype=torch.long, device=self.device)
                    k_logits = gen(k_output)
                    k_scores = k_logits.log_softmax(dim=-1)
                    loss_dict["loss"] += self.criterions[k](k_scores, k_gtruth)
                    self._multi_sta(k, k_scores, k_gtruth)
                # temp
            scores = bottled_output
        else:
            scores = self.generators["generator"](bottled_output)
            scores = torch.log_softmax(scores, dim=-1)
            gtruth = target.view(-1)
            loss_dict["loss"] = self.criterion(scores, gtruth)
            if self.lambda_coverage != 0.0:
                coverage_loss = self._compute_coverage_loss(
                    std_attn=std_attn, coverage_attn=coverage_attn)
                loss_dict["loss"] += coverage_loss
            # stats = self._stats(loss.clone(), scores, gtruth)
            #
            # return loss, stats
            self._sta(scores, gtruth, tag_scores)

        self.step += 1
        stats = self._stats(loss_dict, scores, gtruth)
        # return sum(list(loss_dict.values())),  stats
        return loss_dict["loss"] + loss_dict["tag_loss"], stats
        # # TODO(yida) temp rl
        # return loss_dict["t_loss"], stats

    def _compute_coverage_loss(self, std_attn, coverage_attn):
        covloss = torch.min(std_attn, coverage_attn).sum()
        covloss *= self.lambda_coverage
        return covloss

    def t_gen_func(self, logits, step_all=3000):
        # if not self.train:
        #     return logits.log_softmax(dim=-1).max(1)[1].float().unsqueeze(-1)
        # if random.random() < (step_all - self.step) / step_all:
        #     return torch.tensor([random.randint(0, 19) for i in range(logits.shape[0])]).float().unsqueeze(-1).to(
        #         self.device)
        # dist = torch.distributions.Multinomial(logits=logits.log_softmax(dim=-1), total_count=1)
        # return torch.argmax(dist.sample(), dim=1, keepdim=True).float()
        # return logits * 2 + 1e-4
        probs = (logits * 1).softmax(dim=-1)
        index = torch.arange(1, probs.shape[-1] + 1, dtype=torch.float, device=self.device).unsqueeze(-1)
        return torch.mm(probs, index) / 10

    def _sta(self, scores, gtruth, tag_scores):
        if not self.sta:
            return
        # high_low probs
        argmax_scores, argmax_ids = scores.max(1)
        log_prob = scores.gather(-1, gtruth.unsqueeze(-1))
        index = self.high_num + 4
        low_index = ~gtruth.lt(index)
        non_pad = gtruth.ne(self.padding_idx)
        high_index = (~low_index) & non_pad
        # low_index = tag_gtruth.eq(5)
        low_prob = torch.exp(log_prob[low_index])
        arg_low = torch.exp(argmax_scores[low_index])
        high_prob = torch.exp(log_prob[high_index])
        arg_high = torch.exp(argmax_scores[high_index])
        self.writer.add_scalars("sta_probs/low_prob",
                                {"low_prob": low_prob.mean(), "arg_low": arg_low.mean()},
                                self.step)
        self.writer.add_scalars("sta_probs/high_prob",
                                {"high_prob": high_prob.mean(), "arg_high": arg_high.mean()},
                                self.step)
        if "tag" in self.generators:
            # align
            non_padding = gtruth.ne(self.padding_idx)
            preds_words = scores.max(1)[1].lt(index)
            preds_tag = tag_scores.max(1)[1].eq(4)
            num_correct_tag = preds_words.eq(preds_tag).masked_select(non_padding).sum().item()
            self.writer.add_scalars("sta_align",
                                    {"align_acc": 100 * num_correct_tag / non_padding.sum().item()},
                                    self.step)

    def _multi_sta(self, k, k_scores, k_gtruth):
        if not self.sta:
            return
        k_log_prob = k_scores.gather(-1, k_gtruth.unsqueeze(-1))
        k_rank = k_scores.ge(k_log_prob).sum(-1).float()
        k_prob = torch.exp(k_log_prob)
        k_argmax_scores, high_argmax_ids = k_scores.max(1)
        arg_k = torch.exp(k_argmax_scores)
        self.writer.add_scalars("sta_probs/probs_{}".format(k),
                                {"{}_prob".format(k): k_prob.mean(), "arg_{}".format(k): arg_k.mean()}, self.step)
        self.writer.add_scalars("sta_probs/rank_{}".format(k),
                                {"rank_mean".format(k): k_rank.mean(),
                                 "mean+std".format(k): k_rank.mean() + k_rank.std(),
                                 "mean-std".format(k): k_rank.mean() - k_rank.std()},
                                self.step)


def filter_shard_state(state, shard_size=None):
    for k, v in state.items():
        if shard_size is None:
            yield k, v

        if v is not None:
            v_split = []
            if isinstance(v, torch.Tensor):
                for v_chunk in torch.split(v, shard_size):
                    v_chunk = v_chunk.data.clone()
                    v_chunk.requires_grad = v.requires_grad
                    v_split.append(v_chunk)
            yield k, (v, v_split)


def shards(state, shard_size, eval_only=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval_only: If True, only yield the state, nothing else.
              Otherwise, yield shards.

    Yields:
        Each yielded shard is a dict.

    Side effect:
        After the last shard, this function does back-propagation.
    """
    if eval_only:
        yield filter_shard_state(state)
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state, shard_size))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, [v_chunk for v_chunk in v_split])
                             for k, (_, v_split) in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = []
        for k, (v, v_split) in non_none.items():
            if isinstance(v, torch.Tensor) and state[k].requires_grad:
                variables.extend(zip(torch.split(state[k], shard_size),
                                     [v_chunk.grad for v_chunk in v_split]))
        if variables:
            inputs, grads = zip(*variables)
            torch.autograd.backward(inputs, grads)

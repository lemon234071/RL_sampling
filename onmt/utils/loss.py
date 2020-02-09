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
from onmt.modules.sparse_activations import LogSparsemax
from onmt.modules.sparse_losses import SparsemaxLoss


def build_loss_compute(model, tgt_field, opt, train=True):
    """
    Returns a LossCompute subclass which wraps around an nn.Module subclass
    (such as nn.NLLLoss) which defines the loss criterion. The LossCompute
    object allows this loss to be computed in shards and passes the relevant
    data to a Statistics object which handles training/validation logging.
    Currently, the NMTLossCompute class handles all loss computation except
    for when using a copy mechanism.
    """
    device = torch.device("cuda" if onmt.utils.misc.use_gpu(opt) else "cpu")

    padding_idx = tgt_field.vocab.stoi[tgt_field.pad_token]
    unk_idx = tgt_field.vocab.stoi[tgt_field.unk_token]

    if opt.lambda_coverage != 0:
        assert opt.coverage_attn, "--coverage_attn needs to be set in " \
                                  "order to use --lambda_coverage != 0"

    if opt.copy_attn:
        criterion = onmt.modules.CopyGeneratorLoss(
            len(tgt_field.vocab), opt.copy_attn_force,
            unk_index=unk_idx, ignore_index=padding_idx
        )
    elif opt.label_smoothing > 0 and train:
        criterion = LabelSmoothingLoss(
            opt.label_smoothing, len(tgt_field.vocab), ignore_index=padding_idx
        )
    elif isinstance(model.generator[-1], LogSparsemax):
        criterion = SparsemaxLoss(ignore_index=padding_idx, reduction='sum')
    else:
        criterion = nn.NLLLoss(ignore_index=padding_idx, reduction='sum')

    # if the loss function operates on vectors of raw logits instead of
    # probabilities, only the first part of the generator needs to be
    # passed to the NMTLossCompute. At the moment, the only supported
    # loss function of this kind is the sparsemax loss.
    use_raw_logits = isinstance(criterion, SparsemaxLoss)
    loss_gen = model.generator[0] if use_raw_logits else model.generator
    # TODO(yida) loss
    if opt.copy_attn:
        compute = onmt.modules.CopyGeneratorLossCompute(
            criterion, loss_gen, tgt_field.vocab, opt.copy_loss_by_seqlength,
            lambda_coverage=opt.lambda_coverage
        )
    else:
        # TODO(yida) loss
        compute = NMTLossCompute(
            criterion, loss_gen,
            model.tag_generator, model.low_generator, model.t_generator, model.low_t_generator,
            opt.statistic, opt.high_rate, device, train=train,
            lambda_coverage=opt.lambda_coverage)
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
            loss, loss_t, stats = self._compute_loss(batch, **shard_state)
            return loss / float(normalization), stats
        batch_stats = onmt.utils.Statistics()
        for shard in shards(shard_state, shard_size):
            loss, loss_t, stats = self._compute_loss(batch, **shard)
            loss_t.div(float(normalization)).backward()
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
                                     loss["t_loss"].clone().item(),
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
                 tag_generator, low_generator, t_generator, low_t_generator, sta, high_rate,
                 device,
                 train=False,
                 normalization="sents",
                 lambda_coverage=0.0):
        super(NMTLossCompute, self).__init__(criterion, generator)
        self.lambda_coverage = lambda_coverage
        self.tag_generator = tag_generator
        self.low_generator = low_generator
        self.t_generator = t_generator
        self.low_t_generator = low_t_generator
        self.high_rate = high_rate
        self.sta = sta
        self.writer = SummaryWriter(comment="sta_train") if train else SummaryWriter(comment="sta_valid")
        self.step = 0
        self.device = device

    def _make_shard_state(self, batch, output, range_, attns=None, rnn_outs=None):
        # TODO(yida) loss
        if self.tag_generator is not None:
            shard_state = {
                "output": output,
                "target": batch.tgt[range_[0] + 1: range_[1], :, 0],
                "tag_output": rnn_outs if rnn_outs is not [] else output.clone(),
                "tag_target": batch.pos_tgt[range_[0] + 1: range_[1], :, 0]
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

    def _compute_loss(self, batch, output, target, rnn_out=None, tag_output=None, tag_target=None, std_attn=None,
                      coverage_attn=None):
        # TODO(yida) loss
        loss_dict = {"loss": torch.tensor(0.0).to(self.device),
                     "tag_loss": torch.tensor(0.0).to(self.device),
                     "t_loss": torch.tensor(0.0).to(self.device)}
        temp_loss_t = torch.tensor(0.0).to(self.device)
        bottled_output = self._bottle(output)
        gtruth = target.view(-1)
        tag_scores = None

        if self.tag_generator is not None:
            tag_bottled_output = self._bottle(tag_output)
            tag_scores = self.tag_generator(tag_bottled_output)
            tag_gtruth = tag_target.view(-1)
            loss_dict["tag_loss"] = self.criterion(tag_scores, tag_gtruth)
            # sta
            tag_pred_indices = tag_scores.max(1)[1]
            tag_non_padding = tag_gtruth.ne(self.padding_idx)
            num_correct_tag = tag_pred_indices.eq(tag_gtruth).masked_select(tag_non_padding).sum().item()
            num_non_padding_tag = tag_non_padding.sum().item()
            self.writer.add_scalars("sta_acc",
                                    {"acc": 100 * num_correct_tag / num_non_padding_tag},
                                    self.step)
        # for multi
        if self.low_generator is not None:
            # pred_tag_index = tag_scores.max(1)[1]
            # high_index = pred_tag_index.eq(4)
            high_gt_indices = tag_gtruth.eq(4)
            low_gt_indices = tag_gtruth.eq(5)
            # unk_indices = gtruth.eq(0)
            # high_gt_indices = unk_indices | high_gt_indices
            # low_gt_indices = low_gt_indices ^ unk_indices
            high_output = bottled_output[high_gt_indices]
            low_output = bottled_output[low_gt_indices]
            high_gt = gtruth[high_gt_indices]
            low_gt = gtruth[low_gt_indices] - self.generator._modules["0"].out_features
            high_scores, low_scores = None, None
            if high_output.shape[0] > 0:
                high_scores = self.generator(high_output)
                if self.t_generator is not None:
                    high_t_input = high_scores.clone().detach()
                    logits_t = self.t_generator(high_t_input)
                    indices = logits_t.max(1)[1]
                    temp_loss_t += self.criterion(logits_t, indices)
                    t = self.t_gen_func(logits_t)
                    t_scores = high_t_input / t
                    t_scores = torch.log_softmax(t_scores, dim=-1)
                    loss_dict["t_loss"] += self.criterion(t_scores, high_gt)
                    self.writer.add_scalars("sta_t/high_t", {"high_t": t.mean()}, self.step)
                high_scores = torch.log_softmax(high_scores, dim=-1)
                loss_dict["loss"] += self.criterion(high_scores, high_gt)
            if low_output.shape[0] > 0:
                low_scores = self.low_generator(low_output)
                if self.low_t_generator is not None:
                    low_t_input = low_scores.clone().detach()
                    logits_t = self.low_t_generator(low_t_input)
                    indices = logits_t.max(1)[1]
                    temp_loss_t += self.criterion(logits_t, indices)
                    t = self.t_gen_func(logits_t)
                    t_scores = low_t_input / t
                    t_scores = torch.log_softmax(t_scores, dim=-1)
                    loss_dict["t_loss"] += self.criterion(t_scores, low_gt)
                    self.writer.add_scalars("sta_t/low_t", {"low_t": t.mean()}, self.step)
                self.writer.add_scalars("sta_t/loss_t", {"loss_t": temp_loss_t.div(output.shape[1])}, self.step)
                low_scores = torch.log_softmax(low_scores, dim=-1)
                loss_dict["loss"] += self.criterion(low_scores, low_gt)
            if self.sta:
                self._multi_sta(high_scores, low_scores, high_gt, low_gt)
            # temp
            scores = bottled_output
        else:
            scores = self.generator(bottled_output)
            if self.t_generator is not None:
                logits_t = self.t_generator(scores)
                t = self.t_gen_func(logits_t)
                t_scores = scores / t
                loss_dict["t_loss"] += self.criterion(t_scores, gtruth)
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
            if self.sta:
                self._sta(scores, gtruth, tag_scores)

        self.step += 1
        stats = self._stats(loss_dict, scores, gtruth)
        # return sum(list(loss_dict.values())),  stats
        return loss_dict["loss"] + loss_dict["tag_loss"], loss_dict["t_loss"], stats

    def _compute_coverage_loss(self, std_attn, coverage_attn):
        covloss = torch.min(std_attn, coverage_attn).sum()
        covloss *= self.lambda_coverage
        return covloss

    def t_gen_func(self, logits):
        probs = (logits * 1e4).softmax(dim=-1)
        index = torch.arange(1, probs.shape[-1] + 1, dtype=torch.float, device=self.device).unsqueeze(-1)
        return torch.mm(probs, index) / 10

    def _sta(self, scores, gtruth, tag_scores):
        # probs
        argmax_scores, argmax_ids = scores.max(1)
        log_prob = scores.gather(-1, gtruth.unsqueeze(-1))
        index = int(self.high_rate * (scores.shape[-1] - 4)) + 4
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
        if self.tag_generator is not None:
            # align
            non_padding = gtruth.ne(self.padding_idx)
            preds_words = scores.max(1)[1].lt(index)
            preds_tag = tag_scores.max(1)[1].eq(4)
            num_correct_tag = preds_words.eq(preds_tag).masked_select(non_padding).sum().item()
            self.writer.add_scalars("sta_align",
                                    {"align_acc": 100 * num_correct_tag / non_padding.sum().item()},
                                    self.step)

    def _multi_sta(self, high_scores, low_scores, high_gt, low_gt):
        if high_scores is not None:
            high_log_prob = high_scores.gather(-1, high_gt.unsqueeze(-1))
            high_argmax_scores, high_argmax_ids = high_scores.max(1)
            high_prob = torch.exp(high_log_prob)
            arg_high = torch.exp(high_argmax_scores)
            self.writer.add_scalars("sta_probs/high_prob",
                                    {"high_prob": high_prob.mean(), "arg_high": arg_high.mean()},
                                    self.step)
        if low_scores is not None:
            low_log_prob = low_scores.gather(-1, low_gt.unsqueeze(-1))
            low_argmax_scores, low_argmax_ids = low_scores.max(1)
            low_prob = torch.exp(low_log_prob)
            arg_low = torch.exp(low_argmax_scores)
            self.writer.add_scalars("sta_probs/low_prob",
                                    {"low_prob": low_prob.mean(), "arg_low": arg_low.mean()},
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
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)

""" Translation main class """
from __future__ import unicode_literals, print_function

import torch

from onmt.inputters.text_dataset import TextMultiField


class TranslationBuilder(object):
    """
    Build a word-based translation from the batch output
    of translator and the underlying dictionaries.

    Replacement based on "Addressing the Rare Word
    Problem in Neural Machine Translation" :cite:`Luong2015b`

    Args:
       data (onmt.inputters.Dataset): Data.
       fields (List[Tuple[str, torchtext.data.Field]]): data fields
       n_best (int): number of translations produced
       replace_unk (bool): replace unknown words using attention
       has_tgt (bool): will the batch have gold targets
    """

    def __init__(self, data, fields, n_best=1, replace_unk=False,
                 has_tgt=False, phrase_table=""):
        self.data = data
        self.fields = fields
        self._has_text_src = isinstance(
            dict(self.fields)["src"], TextMultiField)
        self.n_best = n_best
        self.replace_unk = replace_unk
        self.phrase_table = phrase_table
        self.has_tgt = has_tgt

    def _build_target_tokens(self, src, src_vocab, src_raw, pred, attn):
        tgt_field = dict(self.fields)["tgt"].base_field
        vocab = tgt_field.vocab
        tokens = []
        for tok in pred:
            if tok < len(vocab):
                tokens.append(vocab.itos[tok])
            else:
                tokens.append(src_vocab.itos[tok - len(vocab)])
            if tokens[-1] == tgt_field.eos_token:
                tokens = tokens[:-1]
                break
        if self.replace_unk and attn is not None and src is not None:
            for i in range(len(tokens)):
                if tokens[i] == tgt_field.unk_token:
                    _, max_index = attn[i][:len(src_raw)].max(0)
                    tokens[i] = src_raw[max_index.item()]
                    if self.phrase_table != "":
                        with open(self.phrase_table, "r") as f:
                            for line in f:
                                if line.startswith(src_raw[max_index.item()]):
                                    tokens[i] = line.split('|||')[1].strip()
        return tokens

    # TODO(yida)
    def _build_target_tokens_pos(self, src, src_vocab, src_raw, pred, pos_pred, attn):
        tgt_field = dict(self.fields)["tgt"].base_field
        vocab = tgt_field.vocab
        pos_field = dict(self.fields)["pos_tgt"].base_field
        pos_vocab = pos_field.vocab
        pos_tokens = []
        tokens = []
        for tok, pos_tok in zip(pred, pos_pred):
            if tok < len(vocab):
                tokens.append(vocab.itos[tok])
                pos_tokens.append(pos_vocab.itos[pos_tok])
            else:
                tokens.append(src_vocab.itos[tok - len(vocab)])
            if tokens[-1] == tgt_field.eos_token:
                tokens = tokens[:-1]
                pos_tokens = pos_tokens[:-1]
                break
        if self.replace_unk and attn is not None and src is not None:
            for i in range(len(tokens)):
                if tokens[i] == tgt_field.unk_token:
                    _, max_index = attn[i].max(0)
                    tokens[i] = src_raw[max_index.item()]
                    if self.phrase_table != "":
                        with open(self.phrase_table, "r") as f:
                            for line in f:
                                if line.startswith(src_raw[max_index.item()]):
                                    tokens[i] = line.split('|||')[1].strip()
        return tokens, pos_tokens

    def from_batch(self, translation_batch):
        batch = translation_batch["batch"]
        assert (len(translation_batch["gold_score"]) ==
                len(translation_batch["predictions"]))
        if "pos_predictions" in translation_batch:
            assert (len(translation_batch["gold_score"]) ==
                    len(translation_batch["pos_predictions"]))
        batch_size = batch.batch_size

        preds, pred_score, attn, gold_score, indices = list(zip(
            *sorted(zip(translation_batch["predictions"],
                        translation_batch["scores"],
                        translation_batch["attention"],
                        translation_batch["gold_score"],
                        batch.indices.data),
                    key=lambda x: x[-1])))
        # yida translate
        if "pos_predictions" in translation_batch:
            pos_preds, pos_indices = list(zip(
                *sorted(zip(translation_batch["pos_predictions"],
                            batch.indices.data),
                        key=lambda x: x[-1])))
            assert pos_indices == indices

        # Sorting
        inds, perm = torch.sort(batch.indices)
        if self._has_text_src:
            src = batch.src[0][:, :, 0].index_select(1, perm)
        else:
            src = None
        tgt = batch.tgt[:, :, 0].index_select(1, perm) \
            if self.has_tgt else None

        translations = []
        for b in range(batch_size):
            if self._has_text_src:
                src_vocab = self.data.src_vocabs[inds[b]] \
                    if self.data.src_vocabs else None
                src_raw = self.data.examples[inds[b]].src[0]
                tgt_raw = self.data.examples[inds[b]].tgt[0] if hasattr(self.data.examples[inds[b]], "tgt") else None
            else:
                src_vocab = None
                src_raw = None
                tgt_raw = None

            if "pos_predictions" not in translation_batch:
                pred_sents = [self._build_target_tokens(
                    src[:, b] if src is not None else None,
                    src_vocab, src_raw,
                    preds[b][n], attn[b][n])
                    for n in range(self.n_best)]
            else:
                temp_sents = [self._build_target_tokens_pos(
                    src[:, b] if src is not None else None,
                    src_vocab, src_raw,
                    preds[b][n], pos_preds[b][n],
                    attn[b][n])
                    for n in range(self.n_best)]
                pred_sents, pos_pred_sents = [temp_sents[0][0]], [temp_sents[0][1]]
            gold_sent = None
            if tgt is not None:
                gold_sent = self._build_target_tokens(
                    src[:, b] if src is not None else None,
                    src_vocab, src_raw,
                    tgt[1:, b] if tgt is not None else None, None)

            # yida translate
            if "pos_predictions" not in translation_batch:
                translation = Translation(
                    src[:, b] if src is not None else None,
                    src_raw, tgt_raw, pred_sents, attn[b], pred_score[b],
                    gold_sent, gold_score[b]
                )
            else:
                translation = Translation(
                    src[:, b] if src is not None else None,
                    src_raw, tgt_raw, pred_sents,
                    attn[b], pred_score[b],
                    gold_sent, gold_score[b],
                    pos_pred_sents
                )
            translations.append(translation)

        return translations


class Translation(object):
    """Container for a translated sentence.

    Attributes:
        src (LongTensor): Source word IDs.
        src_raw (List[str]): Raw source words.
        pred_sents (List[List[str]]): Words from the n-best translations.
        pred_scores (List[List[float]]): Log-probs of n-best translations.
        attns (List[FloatTensor]) : Attention distribution for each
            translation.
        gold_sent (List[str]): Words from gold translation.
        gold_score (List[float]): Log-prob of gold translation.
    """

    __slots__ = ["src", "src_raw", "pred_sents", "attns", "pred_scores",
                 # yida translate
                 "pos_pred_sents", "tgt_raw",
                 "gold_sent", "gold_score"]

    def __init__(self, src, src_raw, tgt_raw, pred_sents,
                 attn, pred_scores, tgt_sent, gold_score,
                 # yida translate
                 pos_pred_sents=None):
        self.src = src
        self.src_raw = src_raw
        self.tgt_raw = tgt_raw
        self.pred_sents = pred_sents
        self.attns = attn
        self.pred_scores = pred_scores
        self.gold_sent = tgt_sent
        self.gold_score = gold_score
        # yida translate
        self.pos_pred_sents = pos_pred_sents

    def log(self, sent_number):
        """
        Log translation.
        """

        msg = ['\nSENT {}: {}\n'.format(sent_number, self.src_raw)]

        best_pred = self.pred_sents[0]
        best_score = self.pred_scores[0]
        pred_sent = ' '.join(best_pred)
        msg.append('PRED {}: {}\n'.format(sent_number, pred_sent))
        msg.append("PRED SCORE: {:.4f}\n".format(best_score))

        if self.gold_sent is not None:
            tgt_sent = ' '.join(self.gold_sent)
            msg.append('GOLD {}: {}\n'.format(sent_number, tgt_sent))
            msg.append(("GOLD SCORE: {:.4f}\n".format(self.gold_score)))
        if len(self.pred_sents) > 1:
            msg.append('\nBEST HYP:\n')
            for score, sent in zip(self.pred_scores, self.pred_sents):
                msg.append("[{:.4f}] {}\n".format(score, sent))

        return "".join(msg)

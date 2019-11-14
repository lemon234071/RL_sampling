""" Onmt NMT Model base class definition """
import torch.nn as nn


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder, decoder, pos_gen):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        # TODO(yida) model
        self.pos_gen = pos_gen

    def forward(self, src, tgt, pos_src, pos_tgt, lengths, bptt=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence of size ``(tgt_len, batch)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        tgt = tgt[:-1]  # exclude last target from inputs
        # TODO(yida) model
        pos_tgt = pos_tgt[:-1] if self.pos_gen else None

        # TODO(yida) model
        pos_align_src = pos_src if self.pos_gen else None
        enc_state, memory_bank, lengths = self.encoder(src, pos_align_src, lengths)

        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)
        # TODO(yida) model
        pos_align_tgt = pos_tgt if self.pos_gen else None
        dec_out, attns = self.decoder(tgt, memory_bank, pos_align_tgt,
                                      memory_lengths=lengths)
        return dec_out, attns

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)

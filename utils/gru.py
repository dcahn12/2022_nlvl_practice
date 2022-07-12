import torch.nn as nn
from utils.rnns import feed_forward_rnn

class GRU(nn.Module):

    def __init__(self, cfg):
        super(GRU, self).__init__()
        self.input_size   = cfg.DYNAMIC_FILTER.GRU.INPUT_SIZE
        self.num_layers   = cfg.DYNAMIC_FILTER.GRU.NUM_LAYERS
        self.hidden_size  = cfg.DYNAMIC_FILTER.GRU.HIDDEN_SIZE
        self.bias         = cfg.DYNAMIC_FILTER.GRU.BIAS
        self.dropout      = cfg.DYNAMIC_FILTER.GRU.DROPOUT
        self.bidirectional= cfg.DYNAMIC_FILTER.GRU.BIDIRECTIONAL
        self.batch_first  = cfg.DYNAMIC_FILTER.GRU.BATCH_FIRST

        self.gru = nn.GRU(input_size   = self.input_size,
                            hidden_size  = self.hidden_size,
                            num_layers   = self.num_layers,
                            bias         = self.bias,
                            dropout      = self.dropout,
                            bidirectional= self.bidirectional,
                            batch_first = self.batch_first)

    def forward(self, sequences, lengths):
        if lengths is None:
            raise "ERROR in this tail you need lengths of sequences."
        return feed_forward_rnn(self.gru, sequences, lengths=lengths)


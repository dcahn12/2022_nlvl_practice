from yacs.config import CfgNode as CN

_C = CN()

_C.GRU = CN()

_C.GRU.INPUT_SIZE = 300
_C.GRU.NUM_LAYERS = 2
_C.GRU.HIDDEN_SIZE = 512
_C.GRU.BIAS = True
_C.GRU.DROPOUT = 0.
_C.GRU.BIDIRECTIONAL = True
_C.GRU.BATCH_FIRST = True

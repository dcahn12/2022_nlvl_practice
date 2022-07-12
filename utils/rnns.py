
import torch
import numpy as np
import torch.nn as nn


def feed_forward_rnn(rnn, embedded_sequence_batch, lengths=None, hidden_tuple=None):
    """
    Recursive function to encapsulate RNN calls.
    :param rnn:
    :param embedded_sequence_batch:
    :param lengths:
    :param hidden_tuple:
    :return:
    """
    if lengths is not None:
        rnn_input, indices_unsort = pack_rnn_input(embedded_sequence_batch, lengths)
        rnn_output, hidden_tuple = rnn(rnn_input, hidden_tuple)
        output = unpack_rnn_output(rnn_output, indices_unsort)
    else:
        output, hidden_tuple = rnn(embedded_sequence_batch, hidden_tuple)

    return output, hidden_tuple


def pad_sequence(sequence, batch_first=True):

    lengths = []
    for s in sequence:
        lengths.append(s.shape[0])
    lengths = np.array(lengths, dtype=np.float32)
    lengths = torch.from_numpy(lengths)

    return nn.utils.rnn.pad_sequence(sequence, batch_first=batch_first), lengths

def pack_rnn_input(embedded_sequence_batch, sequence_lengths):
    '''
    :param embedded_sequence_batch: torch.Tensor(batch_size, seq_len)
    :param sequence_lengths: list(batch_size)
    :return:
    '''
    sequence_lengths = sequence_lengths.cpu().numpy()

    sorted_sequence_lengths = np.sort(sequence_lengths)[::-1]
    sorted_sequence_lengths = torch.from_numpy(sorted_sequence_lengths.copy())

    idx_sort = np.argsort(-sequence_lengths)
    idx_unsort = np.argsort(idx_sort)

    idx_sort = torch.from_numpy(idx_sort)
    idx_unsort = torch.from_numpy(idx_unsort)

    if embedded_sequence_batch.is_cuda:
        idx_sort = idx_sort.cuda()
        idx_unsort = idx_unsort.cuda()

    embedded_sequence_batch = embedded_sequence_batch.index_select(0, idx_sort)

    # # go back to ints as requested by torch (will change in torch 0.4)
    # int_sequence_lengths = [int(elem) for elem in sorted_sequence_lengths.tolist()]
    # Handling padding in Recurrent Networks
    packed_rnn_input = nn.utils.rnn.pack_padded_sequence(embedded_sequence_batch,sorted_sequence_lengths,batch_first=True)
    return packed_rnn_input, idx_unsort

def unpack_rnn_output(packed_rnn_output, indices):
    '''
    :param packed_rnn_output: torch object
    :param indices: Variable(LongTensor) of indices to sort output
    :return:
    '''
    encoded_sequence_batch, _ = nn.utils.rnn.pad_packed_sequence(packed_rnn_output,batch_first=True)
    encoded_sequence_batch = encoded_sequence_batch.index_select(0, indices)

    return encoded_sequence_batch

def mean_pooling(batch_hidden_states, batch_lengths):
    '''
    :param batch_hidden_states: torch.Tensor(batch_size, seq_len, hidden_size)
    :param batch_lengths: list(batch_size)
    :return:
    '''

    batch_lengths = batch_lengths.unsqueeze(1)
    pooled_batch = torch.sum(batch_hidden_states, 1)

    pooled_batch = pooled_batch / batch_lengths.expand_as(pooled_batch).float()

    return pooled_batch


def max_pooling(batch_hidden_states, batch_lengths):
    '''
    :param batch_hidden_states: torch.Tensor(batch_size, seq_len, hidden_size)
    :return:
    '''
    pooled_batch, _ = torch.max(batch_hidden_states, 1)
    return pooled_batch
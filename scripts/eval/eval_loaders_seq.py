import sys
import numpy as np
from ...fhvae.datasets.seq_dataset import NumpyDataset

np.random.seed(123)


def load_data(name, set_name, batch_size):
    root = "./data/%s" % name
    mvn_path = "%s/mvn.pkl" % root
    Dataset = NumpyDataset

    dt_dset = Dataset(
        "%s/%s/feats.scp" % (root, set_name),
        "%s/%s/len.scp" % (root, set_name), preload=False, mvn_path=mvn_path)

    dt_iterator = _load(dt_dset, batch_size)
    return dt_iterator


def pad_feats(feats, seq_lens):
    batch = np.zeros((len(feats), max(seq_lens), feats[0].shape[-1]))
    for i in range(len(feats)):
        batch[i, :seq_lens[i]] = feats[i]

    return batch


def _load(dt_dset, batch_size):
    def _make_batch(seqs, feats, seq_lens, seq2idx):
        x = pad_feats(np.asarray(feats), seq_lens)
        y = np.asarray([seq2idx[seq] for seq in seqs])
        n = np.asarray(seq_lens)
        sorted_indices = np.argsort(n)[::-1]
        x = x[sorted_indices]
        y = y[sorted_indices]
        n = n[sorted_indices]
        return x, y, n

    def dt_iterator(bs=batch_size):
        seq2idx = dict([(seq, i) for i, seq in enumerate(dt_dset.seqlist)])
        _iterator = dt_dset.iterator(bs, shuffle=False, rem=True)
        for seqs, feats, seq_lens, _, _ in _iterator:
            yield _make_batch(seqs, feats, seq_lens, seq2idx)

    return dt_iterator

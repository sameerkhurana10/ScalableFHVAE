import sys
import numpy as np
from ...fhvae.datasets.seg_dataset import NumpyDataset


def load_data(name, batch_size):
    root = "./data/%s" % name
    mvn_path = "%s/train/mvn.pkl" % root
    Dataset = NumpyDataset

    tr_dset = Dataset(
        "%s/train/feats.scp" % root, "%s/train/len.scp" % root, preload=False, mvn_path=mvn_path)
    dt_dset = Dataset(
        "%s/dev/feats.scp" % root, "%s/dev/len.scp" % root, preload=False, mvn_path=mvn_path)

    return _load(tr_dset, dt_dset, batch_size)


def pad_feats(feats, seq_lens):
    batch = np.zeros((len(feats), max(seq_lens), feats[0].shape[-1]))
    for i in range(len(feats)):
        batch[i, :seq_lens[i]] = feats[i]

    return batch


def _load(tr_dset, dt_dset, batch_size):
    def _make_batch(seqs, feats, seq_lens, seq2idx):
        x = pad_feats(np.asarray(feats), seq_lens)
        y = np.asarray([seq2idx[seq] for seq in seqs])
        n = np.asarray(seq_lens)
        sorted_indices = np.argsort(n)[::-1]
        x = x[sorted_indices]
        y = y[sorted_indices]
        n = n[sorted_indices]
        return x, y, n

    tr_nseqs = len(tr_dset.seqlist)
    tr_shape = tr_dset.get_shape()
    tr_size = tr_nseqs
    dt_size = len(dt_dset.seqlist)

    def tr_iterator(bs=batch_size):
        seq2idx = dict([(seq, i) for i, seq in enumerate(tr_dset.seqlist)])
        _iterator = tr_dset.iterator(bs, shuffle=True, rem=False)
        for seqs, feats, seq_lens, _, _ in _iterator:
            yield _make_batch(seqs, feats, seq_lens, seq2idx)

    def dt_iterator(bs=batch_size):
        seq2idx = dict([(seq, i) for i, seq in enumerate(dt_dset.seqlist)])
        _iterator = dt_dset.iterator(bs, shuffle=False, rem=True)
        for seqs, feats, seq_lens, _, _ in _iterator:
            yield _make_batch(seqs, feats, seq_lens, seq2idx)

    return tr_nseqs, tr_shape, tr_size, dt_size, tr_iterator, dt_iterator

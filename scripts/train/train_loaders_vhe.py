import sys
import numpy as np
from ...fhvae.datasets.seq_dataset import KaldiDataset, NumpyDataset


def load_data(name, seg_len, batch_size, is_numpy):
    root = "./data/%s" % name
    mvn_path = "%s/train/mvn.pkl" % root
    Dataset = NumpyDataset if is_numpy else KaldiDataset
    
    tr_dset = Dataset(
            "%s/train/feats.scp" % root, "%s/train/len.scp" % root,
            min_len=seg_len, preload=False, mvn_path=mvn_path)
    dt_dset = Dataset(
            "%s/dev/feats.scp" % root, "%s/dev/len.scp" % root,
            min_len=seg_len, preload=False, mvn_path=mvn_path)

    return _load(tr_dset, dt_dset, seg_len, 21, batch_size)


def _load(tr_dset, dt_dset, seg_len, nseg, batch_size):
    def _make_batch(seqs, feats, lens, seq2idx):
        bsz, time, dim = len(feats), seg_len, feats[0].shape[1]
        x = np.zeros((bsz, time, dim))
        # other segments from the same class as x
        x_support = np.zeros((bsz, nseg-1, time, dim))
        for j, feat in enumerate(feats):
            # segment the feat into seg len long segments
            l = feat.shape[0]
            # choose nseg segments randomly
            starts = np.random.choice(range(l - seg_len + 1), nseg)
            # construct x and x_support
            for i, start in enumerate(starts):
                end = start + seg_len
                # the first segment goes into x, while the rest go into support
                if i == 0:
                    x[j] = feat[start:end]
                else:
                    x_support[j, i-1] = feat[start:end]
        y = np.asarray([seq2idx[seq] for seq in seqs])
        n = np.asarray(lens)
        return x, x_support, y, n

    tr_nseqs = len(tr_dset.seqlist)
    tr_shape = tr_dset.get_shape()
    tr_size = tr_nseqs
    dt_size = len(dt_dset.seqlist)

    def tr_iterator(bs=batch_size):
        seq2idx = dict([(seq, i) for i, seq in enumerate(tr_dset.seqlist)])
        _iterator = tr_dset.iterator(bs, shuffle=True)
        for seqs, feats, lens, _, _ in _iterator:
            yield _make_batch(seqs, feats, lens, seq2idx)
    
    def dt_iterator(bs=batch_size):
        seq2idx = dict([(seq, i) for i, seq in enumerate(dt_dset.seqlist)])
        _iterator = dt_dset.iterator(bs, shuffle=False)
        for seqs, feats, lens, _, _ in _iterator:
            yield _make_batch(seqs, feats, lens, seq2idx)

    return tr_nseqs, tr_shape, tr_size, dt_size, tr_iterator, dt_iterator


if __name__ == "__main__":
    tr_nseqs, tr_shape, tr_size, dt_size, tr_iterator, dt_iterator = load_data("timit", seg_len=20, batch_size=2,
                                                                               is_numpy=True)
    for i, (x, x_sup, y, n) in enumerate(tr_iterator()):
        print(x.shape, x_sup.shape, y, n)
        break

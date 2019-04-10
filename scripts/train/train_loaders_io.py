import sys
import numpy as np
from ...fhvae.datasets.seg_dataset_io import KaldiSegmentDataset, NumpySegmentDataset


def load_data(name, seg_len, seg_shift, batch_size, is_numpy):
    root = "./data/%s" % name
    mvn_path = "%s/train/mvn.pkl" % root
    Dataset = NumpySegmentDataset if is_numpy else KaldiSegmentDataset

    tr_dset = Dataset(
        "%s/train/feats_in.scp" % root, "%s/train/feats_out.scp" % root, "%s/train/len.scp" % root,
        min_len=seg_len, preload=False, mvn_path=mvn_path,
        seg_len=seg_len, seg_shift=seg_shift, rand_seg=True)
    dt_dset = Dataset(
        "%s/dev/feats_in.scp" % root, "%s/dev/feats_out.scp" % root, "%s/dev/len.scp" % root,
        min_len=seg_len, preload=False, mvn_path=mvn_path,
        seg_len=seg_len, seg_shift=seg_len, rand_seg=False)

    return _load(tr_dset, dt_dset, batch_size)


def _load(tr_dset, dt_dset, batch_size):
    def _make_batch(seqs, in_feats, out_feats, nsegs, seq2idx):
        x_in = np.asarray(in_feats)
        x_out = np.asarray(out_feats)
        y = np.asarray([seq2idx[seq] for seq in seqs])
        n = np.asarray(nsegs)
        return x_in, x_out, y, n

    tr_nseqs = len(tr_dset.seqlist)
    tr_shape = tr_dset.get_shape()
    tr_size = tr_dset.size
    dt_size = dt_dset.size

    def tr_iterator(bs=batch_size):
        seq2idx = dict([(seq, i) for i, seq in enumerate(tr_dset.seqlist)])
        _iterator = tr_dset.iterator(bs, seg_shuffle=True, seg_rem=False)
        for seqs, in_feats, out_feats, nsegs, _, _ in _iterator:
            yield _make_batch(seqs, in_feats, out_feats, nsegs, seq2idx)

    def dt_iterator(bs=batch_size):
        seq2idx = dict([(seq, i) for i, seq in enumerate(dt_dset.seqlist)])
        _iterator = dt_dset.iterator(bs, seg_shuffle=False, seg_rem=True)
        for seqs, in_feats, out_feats, nsegs, _, _ in _iterator:
            yield _make_batch(seqs, in_feats, out_feats, nsegs, seq2idx)

    return tr_nseqs, tr_shape, tr_size, dt_size, tr_iterator, dt_iterator

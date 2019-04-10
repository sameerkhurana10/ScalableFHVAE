import sys
import numpy as np
from ...fhvae.datasets.seg_dataset import KaldiSegmentDataset, NumpySegmentDataset


def load_data(name, seg_len, seg_shift, batch_size, lab_specs, talab_specs, is_numpy):
    root = "./data/%s" % name
    mvn_path = "%s/train/mvn.pkl" % root
    Dataset = NumpySegmentDataset if is_numpy else KaldiSegmentDataset
    
    tr_dset = Dataset(
            "%s/train/feats.scp" % root, "%s/train/len.scp" % root,
            min_len=seg_len, preload=False, mvn_path=mvn_path, lab_specs=lab_specs, talab_specs=talab_specs,
            seg_len=seg_len, seg_shift=seg_shift, rand_seg=True)
    dt_dset = Dataset(
            "%s/dev/feats.scp" % root, "%s/dev/len.scp" % root,
            min_len=seg_len, preload=False, mvn_path=mvn_path, lab_specs=lab_specs, talab_specs=talab_specs,
            seg_len=seg_len, seg_shift=seg_len, rand_seg=False)

    return _load(tr_dset, dt_dset, batch_size)


def _load(tr_dset, dt_dset, batch_size):
    def _make_batch(seqs, feats, nsegs, lab, talab, seq2idx):
        x = np.asarray(feats)
        y = np.asarray([seq2idx[seq] for seq in seqs])
        n = np.asarray(nsegs)
        lab = np.asarray(lab)
        talab = np.asarray(talab)
        return x, y, n, lab, talab

    tr_nseqs = len(tr_dset.seqlist)
    tr_shape = tr_dset.get_shape()
    tr_size = tr_dset.size
    dt_size = dt_dset.size

    def tr_iterator(bs=batch_size):
        seq2idx = dict([(seq, i) for i, seq in enumerate(tr_dset.seqlist)])
        _iterator = tr_dset.iterator(bs, lab_names=["spk"], talab_names=["phn"], seg_shuffle=True, seg_rem=False)
        for seqs, feats, nsegs, lab, talab in _iterator:
            yield _make_batch(seqs, feats, nsegs, lab, talab, seq2idx)
    
    def dt_iterator(bs=batch_size):
        seq2idx = dict([(seq, i) for i, seq in enumerate(dt_dset.seqlist)])
        _iterator = dt_dset.iterator(bs, seg_shuffle=False, lab_names=["spk"], talab_names=["phn"], seg_rem=True)
        for seqs, feats, nsegs, lab, talab in _iterator:
            yield _make_batch(seqs, feats, nsegs, lab, talab, seq2idx)

    return tr_nseqs, tr_shape, tr_size, dt_size, tr_iterator, dt_iterator

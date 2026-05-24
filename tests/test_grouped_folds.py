from ml.folds import grouped_temporal_indices


def test_grouped_temporal_indices_keep_whole_races_together():
    groups = []
    for race_no in range(20):
        groups.extend([f'2026/01/01-R{race_no}'] * 3)

    folds = grouped_temporal_indices(groups)
    seen = {}
    for fold_name, idxs in folds.items():
        for i in idxs:
            g = groups[i]
            assert g not in seen or seen[g] == fold_name
            seen[g] = fold_name

    assert all(len(folds[name]) > 0 for name in ('train', 'val', 'calib', 'test'))

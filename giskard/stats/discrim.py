import numpy as np


def calc_discriminability_statistic(dissimilarity, labels, count_ties=False):
    uni_labels = np.unique(labels)

    all_sample_discrims = []
    for i, row in enumerate(dissimilarity):
        same_mask = labels == labels[i]
        diff_mask = ~same_mask
        same_mask[i] = False
        dist_same_label = row[same_mask]
        dist_diff_label = row[diff_mask]
        sample_discrim = np.less(
            dist_same_label[:, None], dist_diff_label[None, :]
        ).mean()
        if count_ties:
            # Takes roughly double the time to check, true ties are quite rare in most
            # cases, if dissimilarities are discrete that might not be true
            sample_discrim += (
                0.5
                * np.equal(dist_same_label[:, None], dist_diff_label[None, :]).mean()
            )
        all_sample_discrims.append(sample_discrim)
    all_sample_discrims = np.array(all_sample_discrims)

    total_discrim = np.mean(all_sample_discrims)

    class_discrim = {}
    for ul in uni_labels:
        class_discrim[ul] = np.mean(all_sample_discrims[labels == ul])
    return total_discrim, class_discrim

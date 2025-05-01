import numpy as np


class HierarchicalHistogram:
    def __init__(self, hier_hist):
        self.data = hier_hist
        self.height = len(hier_hist) - 1
        self.branch = len(hier_hist[1])
        self.N = hier_hist[0][0]
        self.n_leaf = self.branch**self.height

    def quantile(self, q_frac):
        # Reference: method `get_quantile_estimate` in class `FOQuant`
        # https://figshare.com/s/607998e479b0778645f6
        tol = 1 / (2 * self.branch**self.height)

        lb, ub = 0, 1
        q = 0
        idx = 0

        for level in range(1, self.height + 1):
            interval = (ub - lb) / self.branch
            level_count = sum(self.data[level])
            bin_frac = max(self.data[level][idx] / level_count, 0.0)

            child_idx = self.branch * idx
            max_idx = idx + self.branch - 1

            while q + bin_frac < q_frac and idx < max_idx:
                q += bin_frac
                child_idx += self.branch
                idx += 1
                lb += interval
                bin_frac = max(self.data[level][idx] / level_count, 0.0)

            ub = lb + interval
            idx = child_idx

            if abs(q_frac - q) < tol:
                break

        return lb

    def n_quantile(self, q_list):
        q_list = np.asarray(q_list)
        return np.asarray([self.quantile(q_frac) for q_frac in q_list])

    def merge(self, other: "HierarchicalHistogram"):
        if self.height != other.height:
            raise RuntimeError("Hierarchical histograms must have the same height")
        if self.branch != other.branch:
            raise RuntimeError("Hierarchical histograms must have the same branch")

        merged_data = [
            np.asarray(h1) + np.asarray(h2) for h1, h2 in zip(self.data, other.data)
        ]
        return HierarchicalHistogram(merged_data)

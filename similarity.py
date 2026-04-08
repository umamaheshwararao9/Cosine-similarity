import time
import numpy as np


class CosineSimilarity:
    def compute_fast(self, x):
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x_norm = x / np.clip(norms, 1e-12, None)
        return x_norm @ x_norm.T

    def compute_naive(self, x_small):
        n = x_small.shape[0]
        sims = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                a = x_small[i]
                b = x_small[j]
                num = float(np.dot(a, b))
                den = float(np.linalg.norm(a) * np.linalg.norm(b))
                sims[i, j] = num / den if den > 0 else 0.0
        return sims

    def compare_runtime(self, x, fast_n=2000, naive_n=200):
        x_fast = x[: min(fast_n, x.shape[0])]
        x_naive = x[: min(naive_n, x.shape[0])]
        t0 = time.time()
        self.compute_fast(x_fast)
        fast_time = time.time() - t0
        t1 = time.time()
        self.compute_naive(x_naive)
        naive_time = time.time() - t1
        scale = (x_fast.shape[0] / x_naive.shape[0]) ** 2
        est_naive_fast_n = naive_time * scale
        speedup = est_naive_fast_n / max(fast_time, 1e-12)
        return {"fast_time": fast_time, "naive_time": naive_time, "speedup": speedup}

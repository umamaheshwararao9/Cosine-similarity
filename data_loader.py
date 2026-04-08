import numpy as np
import pandas as pd


class DataLoader:
    def __init__(self, n=10000, d=300, seed=42):
        self.n = n
        self.d = d
        self.seed = seed

    def _load_online(self):
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
        try:
            df = pd.read_csv(url)
            arr = df.select_dtypes(include="number").to_numpy(dtype=np.float32)
            if arr.size == 0:
                return None
            reps = int(np.ceil(self.n / arr.shape[0]))
            tiled = np.tile(arr, (reps, int(np.ceil(self.d / arr.shape[1]))))
            return tiled[: self.n, : self.d]
        except Exception:
            return None

    def get_data(self, use_online=False):
        if use_online:
            x = self._load_online()
            if x is not None:
                return x
        rng = np.random.default_rng(self.seed)
        return rng.normal(size=(self.n, self.d)).astype(np.float32)

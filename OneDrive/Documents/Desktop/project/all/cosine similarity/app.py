import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import DataLoader
from similarity import CosineSimilarity


st.title("Cosine Similarity Visualization")
subset = st.slider("Heatmap subset size", 20, 100, 50, 5)
online = st.checkbox("Try online dataset first", value=False)

loader = DataLoader()
x = loader.get_data(use_online=online)
cos = CosineSimilarity()
times = cos.compare_runtime(x)
sim_subset = cos.compute_fast(x[:subset])

st.write("Dataset shape:", x.shape)
st.write("Similarity subset shape:", sim_subset.shape)
st.write(
    pd.DataFrame(
        {
            "Metric": ["Fast method time (s)", "Naive method time (s)", "Speedup factor"],
            "Value": [times["fast_time"], times["naive_time"], times["speedup"]],
        }
    )
)
st.write("Similarity matrix (first 10x10 view):")
st.dataframe(pd.DataFrame(sim_subset[:10, :10]))

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(sim_subset, cmap="viridis", aspect="auto")
ax.set_title(f"Cosine Similarity Heatmap ({subset}x{subset})")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
st.pyplot(fig)

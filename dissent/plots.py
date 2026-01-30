# Standard library
from pathlib import Path

# Third-party libraries
import numpy as np
import pandas as pd
import typer
from loguru import logger
from tqdm import tqdm
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import plotly.graph_objects as go

# Local application imports
from dissent.config import FIGURES_DIR, PROCESSED_DATA_DIR, MODELS_DIR

app = typer.Typer()


@app.command()
def main(): 
    model = MODELS_DIR / "word2vec.model"

    # ----- GLOBAL PCA -----
    all_words = model.wv.index_to_key
    all_vectors = np.array([model.wv[word] for word in all_words])

    pca = PCA(n_components=2, random_state=42)
    all_proj = pca.fit_transform(all_vectors)

    df_global = pd.DataFrame({
        "x": all_proj[:, 0],
        "y": all_proj[:, 1],
        "word": all_words
    })

    # ----- SEMANTIC CORNER -----
    anchors = ["political", "judge", "court"]
    top_n = 30

    words_to_plot = set(anchors)
    for anchor in anchors:
        for word, _ in model.wv.most_similar(anchor, topn=top_n):
            words_to_plot.add(word)

    words_to_plot = list(words_to_plot)
    corner_vectors = np.array([model.wv[word] for word in words_to_plot])
    corner_proj = pca.transform(corner_vectors)

    df_corner = pd.DataFrame({
        "x": corner_proj[:, 0],
        "y": corner_proj[:, 1],
        "word": words_to_plot
    })

    # Bounding box for corner in global space
    x_min, x_max = df_corner["x"].min(), df_corner["x"].max()
    y_min, y_max = df_corner["y"].min(), df_corner["y"].max()

    # ----- FIGURE -----
    fig = go.Figure()

    # MAIN VIEW: Semantic corner (zoomed)
    fig.add_trace(go.Scatter(
        x=df_corner["x"],
        y=df_corner["y"],
        mode="markers+text",
        text=df_corner["word"],
        textposition="top center",
        marker=dict(size=9),
        name="Semantic Corner"
    ))

    # INSET: Global space (mini-map)
    fig.add_trace(go.Scatter(
        x=df_global["x"],
        y=df_global["y"],
        mode="markers",
        marker=dict(size=2, opacity=0.3),
        name="Global Space",
        xaxis="x2",
        yaxis="y2",
        hoverinfo="skip"
    ))

    # Bounding box on inset
    fig.add_shape(
        type="rect",
        x0=x_min,
        x1=x_max,
        y0=y_min,
        y1=y_max,
        xref="x2",
        yref="y2",
        line=dict(width=2),
    )

    # Layout: define inset axes
    fig.update_layout(
        title="Semantic Corner with Global Inset Map (PCA)",
        width=900,
        height=900,

        xaxis=dict(domain=[0, 1], title="PCA 1"),
        yaxis=dict(domain=[0, 1], title="PCA 2"),

        xaxis2=dict(domain=[0.70, 0.98], anchor="y2", showgrid=False, title=""),
        yaxis2=dict(domain=[0.70, 0.98], anchor="x2", showgrid=False, title=""),

        showlegend=False
    )
    fig.write_html("semantic_corner_plot.html")
    fig.write_image("semantic_corner_plot.png", scale = 4)
    fig.show()

if __name__ == "__main__":
    app()

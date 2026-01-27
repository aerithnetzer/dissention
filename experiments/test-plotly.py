import plotly.graph_objects as go
import numpy as np

# Create some random data
np.random.seed(42)
x = np.random.randn(500)
y = np.random.randn(500)

# Main scatter plot (zoomed-in view)
fig = go.Figure()

fig.add_trace(
    go.Scatter(x=x, y=y, mode="markers", marker=dict(color="blue", size=5), name="Zoomed view")
)

# Inset scatter plot (full overview)
fig.add_trace(
    go.Scatter(
        x=x,
        y=y,
        mode="markers",
        marker=dict(color="lightgrey", size=3),
        name="Overview",
        xaxis="x2",
        yaxis="y2",
    )
)

# Define inset axes
fig.update_layout(
    xaxis=dict(domain=[0, 1], range=[-2, 2]),  # main plot x
    yaxis=dict(domain=[0, 1], range=[-2, 2]),  # main plot y
    xaxis2=dict(domain=[0.1, 0.2], anchor="y2"),  # inset position
    yaxis2=dict(domain=[0.1, 0.2], anchor="x2"),  # inset position
    showlegend=False,
)

# Draw rectangle showing zoomed area on inset
fig.add_shape(
    type="rect",
    x0=-0.5,
    y0=-0.5,
    x1=0.5,
    y1=0.5,  # zoomed-in range
    xref="x2",
    yref="y2",
    line=dict(color="red", width=2),
)

fig.show()


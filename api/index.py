import numpy as np
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go

# -----------------------------------------
# Dash app (MathJax включается флагом mathjax=True в dcc.Markdown)
# -----------------------------------------
dash_app = dash.Dash(__name__)
dash_app.title = "Cosine Similarity — Demo"
server = dash_app.server  # WSGI для Vercel

# ---------- математика ----------
def cosine_block(a, b):
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])

    dot = ax * bx + ay * by
    na = float(np.sqrt(ax * ax + ay * ay))
    nb = float(np.sqrt(bx * bx + by * by))

    cos = dot / (na * nb) if na != 0 and nb != 0 else np.nan
    theta = float(np.degrees(np.arccos(np.clip(cos, -1, 1)))) if not np.isnan(cos) else np.nan

    # Формулы как LaTeX в Markdown. ВАЖНО: используем $$...$$
    md = rf"""
**Векторы**
$$
\mathbf{{A}}=\begin{{bmatrix}}{ax:.2f}\\ {ay:.2f}\end{{bmatrix}},\quad
\mathbf{{B}}=\begin{{bmatrix}}{bx:.2f}\\ {by:.2f}\end{{bmatrix}}
$$

**Скалярное произведение**
$$
\mathbf{{A}}\cdot\mathbf{{B}} = {ax:.2f}\cdot{bx:.2f} + {ay:.2f}\cdot{by:.2f} = {dot:.2f}
$$

**Нормы**
$$
\lVert \mathbf{{A}} \rVert = \sqrt{{{ax:.2f}^2 + {ay:.2f}^2}} = {na:.2f}
\qquad
\lVert \mathbf{{B}} \rVert = \sqrt{{{bx:.2f}^2 + {by:.2f}^2}} = {nb:.2f}
$$

**Косинусное сходство**
$$
\cos(\theta) = \frac{{\mathbf{{A}}\cdot\mathbf{{B}}}}{{\lVert\mathbf{{A}}\rVert\,\lVert\mathbf{{B}}\rVert}}
= {"" if np.isnan(cos) else f"{cos:.3f}"}
$$

**Угол**
$$
\theta = {"" if np.isnan(theta) else f"{theta:.1f}\\,^\\circ"}
$$
"""
    panel = dcc.Markdown(
        md,
        mathjax=True,
        style={
            "fontSize": "16px",
            "lineHeight": "1.45",
            "background": "#fafafa",
            "border": "1px solid #e7e7e7",
            "borderRadius": "10px",
            "padding": "14px",
            "boxShadow": "0 1px 2px rgba(0,0,0,0.06)",
        },
    )
    return cos, theta, dot, na, nb, panel


# ---------- начальные векторы ----------
v1 = np.array([7.0, 3.0])
v2 = np.array([3.0, 7.0])


# ---------- построение графика ----------
def make_figure(a, b):
    ax, ay = a
    bx, by = b

    fig = go.Figure()

    # основные векторы
    fig.add_shape(type="line", x0=0, y0=0, x1=ax, y1=ay, line=dict(color="#e53935", width=5))
    fig.add_shape(type="line", x0=0, y0=0, x1=bx, y1=by, line=dict(color="#1e88e5", width=5))

    # проекции (подвекторы)
    fig.add_trace(go.Scatter(x=[0, ax], y=[0, 0], mode="lines",
                             line=dict(width=3, dash="dash", color="#ef9a9a"), showlegend=False))
    fig.add_trace(go.Scatter(x=[0, 0], y=[0, ay], mode="lines",
                             line=dict(width=3, dash="dash", color="#ef9a9a"), showlegend=False))
    fig.add_trace(go.Scatter(x=[0, bx], y=[0, 0], mode="lines",
                             line=dict(width=3, dash="dash", color="#90caf9"), showlegend=False))
    fig.add_trace(go.Scatter(x=[0, 0], y=[0, by], mode="lines",
                             line=dict(width=3, dash="dash", color="#90caf9"), showlegend=False))

    # маркеры и подписи
    fig.add_trace(go.Scatter(x=[ax], y=[ay], mode="markers+text",
                             marker=dict(size=10, color="#e53935"),
                             text=[f"A({ax:.2f}, {ay:.2f})"], textposition="top center", showlegend=False))
    fig.add_trace(go.Scatter(x=[bx], y=[by], mode="markers+text",
                             marker=dict(size=10, color="#1e88e5"),
                             text=[f"B({bx:.2f}, {by:.2f})"], textposition="top center", showlegend=False))

    # фиксированные оси
    fig.update_layout(
        xaxis=dict(range=[-10, 10], zeroline=True, mirror=True, showgrid=True,
                   gridcolor="#f0f0f0", scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[-10, 10], zeroline=True, mirror=True, showgrid=True, gridcolor="#f0f0f0"),
        margin=dict(l=30, r=20, t=10, b=30),
        width=680, height=680,
        plot_bgcolor="white",
        showlegend=False
    )
    return fig


# ---------- Layout ----------
theory_block = dcc.Markdown(
    """
### Что такое косинусное сходство?

Косинусное сходство измеряет **насколько два вектора направлены в одну сторону**, независимо от их длины.
Если представить векторы как стрелки, оно показывает косинус угла между ними:

- **1** → векторы совпадают по направлению  
- **0** → перпендикулярны  
- **–1** → противоположны

---

### Формула

$$
\\text{cos\\_sim}(A, B) = \\frac{A \\cdot B}{\\|A\\|\\,\\|B\\|}
$$

Эта формула следует из геометрического определения скалярного произведения:

$$
A \\cdot B = \\|A\\|\\,\\|B\\|\\cos(\\theta)
\\quad\\Rightarrow\\quad
\\cos(\\theta) = \\frac{A \\cdot B}{\\|A\\|\\,\\|B\\|}
$$

---

### Почему это полезно

- Убирает влияние длины (нормализация)  
- Смотрит на **направление**, а не на масштаб  
- Легко интерпретировать: ближе к 1 — больше сходство

---

### Пример

Пусть $A=(1,0)$ и $B=(0,1)$.
Тогда $A\\cdot B = 0$, длины равны $1$, а косинусное сходство $=0$ — векторы перпендикулярны.
    """,
    mathjax=True,
    style={
        "background": "#fff",
        "padding": "16px",
        "border": "1px solid #eee",
        "borderRadius": "8px",
        "marginBottom": "20px",
        "fontSize": "15px",
        "lineHeight": "1.6",
    },
)

dash_app.layout = html.Div(
    [
        html.H2("Интерактивная симуляция косинусного сходства (2D)", style={"margin": "10px 0 18px 0"}),

        # Теория сверху
        theory_block,

        # Интерактивная часть
        html.Div(
            [
                dcc.Graph(
                    id="vector-plot",
                    figure=make_figure(v1, v2),
                    config={"editable": True},
                    style={"flex": "2"},
                ),
                html.Div(id="formula-panel", style={"flex": "1", "marginLeft": "20px"}),
            ],
            style={"display": "flex", "flexDirection": "row"},
        ),
    ]
)

# ---------- Callbacks ----------
@dash_app.callback(
    Output("vector-plot", "figure"),
    Output("formula-panel", "children"),
    Input("vector-plot", "relayoutData"),
)
def on_drag(relayoutData):
    global v1, v2

    if relayoutData:
        if "shapes[0].x1" in relayoutData and "shapes[0].y1" in relayoutData:
            v1 = np.array([relayoutData["shapes[0].x1"], relayoutData["shapes[0].y1"]], dtype=float)
        if "shapes[1].x1" in relayoutData and "shapes[1].y1" in relayoutData:
            v2 = np.array([relayoutData["shapes[1].x1"], relayoutData["shapes[1].y1"]], dtype=float)

    cos, theta, dot, na, nb, formulas = cosine_block(v1, v2)
    fig = make_figure(v1, v2)
    subtitle = f"A·B={dot:.2f} | ||A||={na:.2f} ||B||={nb:.2f} | cos(θ)={'' if np.isnan(cos) else f'{cos:.3f}'}"
    fig.update_layout(title=dict(text=subtitle, x=0.5, y=0.97, xanchor="center", yanchor="top", font=dict(size=14)))
    return fig, formulas

# ---------- Экспорт для Vercel ----------
app = server

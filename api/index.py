import numpy as np
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go

# ---------- математика ----------
def cosine_block(a, b):
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])

    dot = ax*bx + ay*by
    na2, nb2 = ax*ax + ay*ay, bx*bx + by*by
    na, nb = np.sqrt(na2), np.sqrt(nb2)

    cos = dot / (na*nb) if na != 0 and nb != 0 else np.nan
    theta = float(np.degrees(np.arccos(np.clip(cos, -1, 1)))) if not np.isnan(cos) else np.nan

    # красивый правый блок с формулами
    def fnum(x, p=2):
        return f"{x:.{p}f}"

    return (
        cos, theta, dot, na, nb,
        html.Div([
            html.H4("Формулы", style={"marginTop": 0, "marginBottom": "8px"}),
            html.Div([
                html.Div([html.B("A = "),
                          html.Code(f"[{fnum(ax)}, {fnum(ay)}]"),
                          " = ", html.Code(f"[{fnum(ax)}, 0]"), " + ", html.Code(f"[0, {fnum(ay)}]")]),
                html.Div([html.B("B = "),
                          html.Code(f"[{fnum(bx)}, {fnum(by)}]"),
                          " = ", html.Code(f"[{fnum(bx)}, 0]"), " + ", html.Code(f"[0, {fnum(by)}]")]),
            ], style={"marginBottom": "10px"}),

            html.Hr(),

            html.Div([
                html.Div([
                    html.B("Скалярное произведение: "),
                    html.Code(f"A · B = {fnum(ax)}·{fnum(bx)} + {fnum(ay)}·{fnum(by)} = {fnum(dot)}")
                ]),
            ], style={"marginBottom": "10px"}),

            html.Div([
                html.Div([
                    html.B("Норма A: "),
                    html.Code(f"||A|| = √({fnum(ax)}² + {fnum(ay)}²) = √({fnum(ax*ax)} + {fnum(ay*ay)}) = {fnum(na)}")
                ]),
                html.Div([
                    html.B("Норма B: "),
                    html.Code(f"||B|| = √({fnum(bx)}² + {fnum(by)}²) = √({fnum(bx*bx)} + {fnum(by*by)}) = {fnum(nb)}")
                ]),
            ], style={"marginBottom": "10px"}),

            html.Div([
                html.B("Косинусное сходство: "),
                html.Code(
                    f"cos(θ) = (A·B) / (||A||·||B||) = {fnum(dot)} / ({fnum(na)}·{fnum(nb)})"
                    + (f" = {fnum(cos,3)}" if not np.isnan(cos) else " = неопределено")
                )
            ], style={"marginBottom": "6px"}),

            html.Div([
                html.B("Угол θ: "),
                html.Code(f"{fnum(theta,1)}°" if not np.isnan(theta) else "неопределён")
            ]),
        ], style={
            "fontSize": "16px",
            "lineHeight": "1.35",
            "background": "#fafafa",
            "border": "1px solid #e7e7e7",
            "borderRadius": "10px",
            "padding": "14px",
            "boxShadow": "0 1px 2px rgba(0,0,0,0.06)"
        })
    )

# ---------- начальные векторы ----------
v1 = np.array([7.0, 3.0])
v2 = np.array([3.0, 7.0])

# ---------- построение графика ----------
def make_figure(a, b):
    ax, ay = a
    bx, by = b

    fig = go.Figure()

    # основные векторы
    fig.add_shape(type="line", x0=0, y0=0, x1=ax, y1=ay,
                  line=dict(color="#e53935", width=5))
    fig.add_shape(type="line", x0=0, y0=0, x1=bx, y1=by,
                  line=dict(color="#1e88e5", width=5))

    # проекции (подвекторы)
    fig.add_trace(go.Scatter(x=[0, ax], y=[0, 0], mode="lines",
                             line=dict(width=3, dash="dash", color="#ef9a9a"),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=[0, 0], y=[0, ay], mode="lines",
                             line=dict(width=3, dash="dash", color="#ef9a9a"),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=[0, bx], y=[0, 0], mode="lines",
                             line=dict(width=3, dash="dash", color="#90caf9"),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=[0, 0], y=[0, by], mode="lines",
                             line=dict(width=3, dash="dash", color="#90caf9"),
                             showlegend=False))

    # маркеры и подписи
    fig.add_trace(go.Scatter(x=[ax], y=[ay], mode="markers+text",
                             marker=dict(size=10, color="#e53935"),
                             text=[f"A({ax:.2f}, {ay:.2f})"], textposition="top center",
                             showlegend=False))
    fig.add_trace(go.Scatter(x=[bx], y=[by], mode="markers+text",
                             marker=dict(size=10, color="#1e88e5"),
                             text=[f"B({bx:.2f}, {by:.2f})"], textposition="top center",
                             showlegend=False))

    # фиксированные оси
    fig.update_layout(
        xaxis=dict(range=[-10, 10], zeroline=True, mirror=True, showgrid=True,
                   gridcolor="#f0f0f0", scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[-10, 10], zeroline=True, mirror=True, showgrid=True,
                   gridcolor="#f0f0f0"),
        margin=dict(l=30, r=20, t=10, b=30),
        width=680, height=680,
        plot_bgcolor="white",
        showlegend=False
    )
    return fig

# ---------- Dash app ----------
app = dash.Dash(__name__)
app.title = "Cosine Similarity — Demo"

# Для работы с Vercel
server = app.server

app.layout = html.Div([
    html.H2("Интерактивная симуляция косинусного сходства (2D)",
            style={"margin": "10px 0 18px 0"}),

    html.Div([
        dcc.Graph(
            id="vector-plot",
            figure=make_figure(v1, v2),
            config={"editable": True},  # тянем концы двух красной/синей линий
            style={"flex": "2"}
        ),
        html.Div(id="formula-panel", style={"flex": "1", "marginLeft": "20px"})
    ], style={"display": "flex", "flexDirection": "row"})
])

@app.callback(
    Output("vector-plot", "figure"),
    Output("formula-panel", "children"),
    Input("vector-plot", "relayoutData")
)
def on_drag(relayoutData):
    global v1, v2

    # перетаскивание shape[0] и shape[1] (концы векторов A и B)
    if relayoutData:
        if "shapes[0].x1" in relayoutData and "shapes[0].y1" in relayoutData:
            v1 = np.array([relayoutData["shapes[0].x1"], relayoutData["shapes[0].y1"]], dtype=float)
        if "shapes[1].x1" in relayoutData and "shapes[1].y1" in relayoutData:
            v2 = np.array([relayoutData["shapes[1].x1"], relayoutData["shapes[1].y1"]], dtype=float)

    cos, theta, dot, na, nb, formulas = cosine_block(v1, v2)
    fig = make_figure(v1, v2)

    # подпишем текущие скалярное и cos прямо на графике
    subtitle = f"A·B={dot:.2f} | ||A||={na:.2f} ||B||={nb:.2f} | cos(θ)={'' if np.isnan(cos) else f'{cos:.3f}'}"
    fig.update_layout(title=dict(text=subtitle, x=0.5, y=0.97, xanchor="center", yanchor="top", font=dict(size=14)))

    return fig, formulas

# Экспорт для Vercel
application = app.server

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)  # Dash 3.x

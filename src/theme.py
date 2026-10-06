"""Dark dashboard theme (GEL-style): CSS + small HTML components for Streamlit."""
import html

import streamlit as st

COLORS = {
    "cyan": "#22d3ee", "red": "#ef4444", "green": "#22c55e", "yellow": "#facc15",
    "orange": "#f97316", "purple": "#8b5cf6", "muted": "#64748b", "text": "#e2e8f0",
}

CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap');
:root{--bg:#0a0e17;--panel:#0f1523;--panel2:#121b2d;--border:#1c2a3f;--text:#e2e8f0;--muted:#64748b;
--cyan:#22d3ee;--red:#ef4444;--green:#22c55e;--yellow:#facc15;--orange:#f97316;--purple:#8b5cf6;
--mono:'JetBrains Mono','SF Mono',Menlo,Consolas,monospace;}
[data-testid="stApp"]{background:var(--bg);}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stSidebar"]{background:#0c1220;border-right:1px solid var(--border);}
.block-container{padding-top:2.2rem;max-width:1450px;}
footer{visibility:hidden;}

.gel-header{display:flex;align-items:center;gap:.6rem;flex-wrap:wrap;margin-bottom:.15rem;}
.gel-bolt{color:var(--cyan);font-size:1.3rem;}
.gel-title{font-family:var(--mono);font-weight:700;font-size:1.5rem;letter-spacing:.04em;color:var(--cyan);}
.gel-pill{font-family:var(--mono);font-size:.68rem;color:var(--cyan);border:1px solid var(--border);
background:var(--panel2);border-radius:999px;padding:.12rem .6rem;letter-spacing:.05em;}
.gel-sub{font-family:var(--mono);font-size:.64rem;color:var(--muted);letter-spacing:.18em;text-transform:uppercase;margin-bottom:1rem;}
.gel-label{font-family:var(--mono);font-size:.62rem;color:var(--muted);letter-spacing:.16em;text-transform:uppercase;margin:.9rem 0 .35rem 0;}

.gel-card{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:.7rem .85rem;height:100%;}
.gel-card .gel-label{margin:0 0 .25rem 0;}
.gel-value{font-family:var(--mono);font-weight:700;font-size:1.35rem;line-height:1.25;}
.gel-sub2{font-size:.72rem;color:var(--muted);margin-top:.2rem;}

.gel-banner{border-radius:8px;padding:.8rem 1rem;margin:.9rem 0;border:1px solid;}
.gel-banner.ok{background:rgba(34,197,94,.08);border-color:rgba(34,197,94,.45);}
.gel-banner.warn{background:rgba(250,204,21,.07);border-color:rgba(250,204,21,.45);}
.gel-banner.danger{background:rgba(239,68,68,.08);border-color:rgba(239,68,68,.5);}
.gel-banner-title{font-family:var(--mono);font-weight:700;font-size:.82rem;letter-spacing:.06em;}
.gel-banner.ok .gel-banner-title{color:var(--green);}
.gel-banner.warn .gel-banner-title{color:var(--yellow);}
.gel-banner.danger .gel-banner-title{color:var(--red);}
.gel-banner-text{font-size:.82rem;color:#cbd5e1;margin-top:.2rem;}

.gel-score{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:1rem .8rem .9rem;text-align:center;}
.gel-score-num{font-family:var(--mono);font-weight:700;font-size:3.4rem;line-height:1;}
.gel-score-of{font-family:var(--mono);font-size:.7rem;color:var(--muted);margin-bottom:.2rem;}
.gel-bar{height:5px;background:var(--border);border-radius:3px;margin:.6rem 0;overflow:hidden;}
.gel-bar div{height:100%;border-radius:3px;}
.gel-caption{font-size:.74rem;color:var(--muted);}
.gel-tiles{display:grid;grid-template-columns:repeat(3,1fr);gap:.4rem;margin:.5rem 0;}
.gel-tile{background:var(--panel);border:1px solid var(--border);border-radius:6px;text-align:center;padding:.45rem .2rem;}
.gel-tile b{display:block;font-family:var(--mono);font-size:1.15rem;}
.gel-tile span{font-family:var(--mono);font-size:.55rem;color:var(--muted);letter-spacing:.12em;text-transform:uppercase;}

.gel-side{background:var(--panel);border:1px solid var(--border);border-left-width:3px;border-radius:6px;padding:.6rem .8rem;margin-bottom:.5rem;}
.gel-side .gel-label{margin:0 0 .15rem 0;}
.gel-side .gel-value{font-size:1.2rem;}
.gel-note{background:var(--panel2);border:1px solid var(--border);border-radius:8px;padding:.6rem .9rem;font-size:.78rem;color:#94a3b8;margin-bottom:.6rem;}

button[data-baseweb="tab"]{font-family:var(--mono);font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);}
button[data-baseweb="tab"][aria-selected="true"]{color:var(--text);}
div[data-baseweb="tab-highlight"]{background-color:var(--red);}
div[data-baseweb="tab-border"]{background-color:var(--border);}

.stButton>button{background:var(--panel);border:1px solid var(--border);color:var(--text);border-radius:8px;font-size:.8rem;}
.stButton>button:hover{border-color:var(--cyan);color:var(--cyan);}
.stButton>button[kind="primary"],.stButton>button[data-testid="stBaseButton-primary"]{background:#0e2238;border:1px solid #1e4a6e;color:var(--cyan);font-family:var(--mono);letter-spacing:.1em;font-weight:700;}
.stButton>button[kind="primary"]:hover,.stButton>button[data-testid="stBaseButton-primary"]:hover{border-color:var(--cyan);color:#fff;}
[data-testid="stMetric"]{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:.6rem .8rem;}
[data-testid="stMetricLabel"] p{font-family:var(--mono);font-size:.64rem;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);}
[data-testid="stExpander"]{background:var(--panel);border:1px solid var(--border);border-radius:8px;}
"""


def esc(value):
    return html.escape(str(value))


def inject_css():
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def render(markup):
    st.markdown(markup, unsafe_allow_html=True)


def header(title, badges, subtitle):
    pills = "".join(f'<span class="gel-pill">{esc(b)}</span>' for b in badges)
    return (f'<div class="gel-header"><span class="gel-bolt">⚡</span><span class="gel-title">{esc(title)}</span>'
            f'{pills}</div><div class="gel-sub">{esc(subtitle)}</div>')


def label(text):
    return f'<div class="gel-label">{esc(text)}</div>'


def card(title, value, sub="", color=COLORS["text"]):
    return (f'<div class="gel-card"><div class="gel-label">{esc(title)}</div>'
            f'<div class="gel-value" style="color:{color}">{esc(value)}</div>'
            f'<div class="gel-sub2">{esc(sub)}</div></div>')


def banner(kind, icon, title, text):
    return (f'<div class="gel-banner {kind}"><div class="gel-banner-title">{icon} {esc(title)}</div>'
            f'<div class="gel-banner-text">{esc(text)}</div></div>')


def score_card(score, title, caption, color):
    score = max(0, min(100, int(round(score))))
    return (f'<div class="gel-score" style="border-top:3px solid {color}">'
            f'<div class="gel-score-num" style="color:{color}">{score}</div><div class="gel-score-of">/100</div>'
            f'<div class="gel-label" style="margin:.3rem 0 0 0">{esc(title)}</div>'
            f'<div class="gel-bar"><div style="width:{score}%;background:{color}"></div></div>'
            f'<div class="gel-caption">{esc(caption)}</div></div>')


def tiles(items):
    cells = "".join(
        f'<div class="gel-tile"><b style="color:{color}">{esc(value)}</b><span>{esc(name)}</span></div>'
        for value, name, color in items
    )
    return f'<div class="gel-tiles">{cells}</div>'


def side_card(title, value, sub, color):
    return (f'<div class="gel-side" style="border-left-color:{color}"><div class="gel-label">{esc(title)}</div>'
            f'<div class="gel-value" style="color:{color}">{esc(value)}</div>'
            f'<div class="gel-sub2">{esc(sub)}</div></div>')


def note(text):
    return f'<div class="gel-note">{esc(text)}</div>'
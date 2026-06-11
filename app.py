import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import date
import os

st.set_page_config(
    page_title="World Cup Predictor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg:        #07070f;
    --surface:   #0d0d1c;
    --surface2:  #121224;
    --border:    #1c1c30;
    --border2:   #2a2a40;
    --text:      #d0d0e8;
    --muted:     #52527a;
    --accent:    #4f9fff;
    --adim:      rgba(79,159,255,0.08);
    --home:      #4f9fff;
    --draw:      #ffcc00;
    --away:      #ff4466;
    --mono:      'IBM Plex Mono', monospace;
    --sans:      'Space Grotesk', sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--sans) !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp { background-color: var(--bg) !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
    min-width: 220px !important;
    max-width: 220px !important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] label {
    color: #d0d0e8 !important;
    font-family: var(--sans) !important;
}
section[data-testid="stSidebar"] hr {
    border-color: var(--border) !important;
}
/* Radio nav labels */
section[data-testid="stSidebar"] .stRadio > div > label {
    font-size: 13px !important;
    font-weight: 500 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 6px 0 !important;
    color: #d0d0e8 !important;
}
section[data-testid="stSidebar"] .stRadio > div {
    gap: 4px !important;
}
/* Radio circle */
section[data-testid="stSidebar"] .stRadio input[type="radio"] + div {
    border-color: #52527a !important;
}
section[data-testid="stSidebar"] .stRadio input[type="radio"]:checked + div {
    border-color: var(--accent) !important;
    background-color: var(--accent) !important;
}

/* Remove default padding */
.block-container { padding: 2rem 2.5rem !important; max-width: 1400px !important; }

/* Headings */
h1 { font-family: var(--sans) !important; font-weight: 700 !important;
     font-size: 2rem !important; letter-spacing: -0.02em !important;
     color: #ffffff !important; margin-bottom: 4px !important; }
h2 { font-family: var(--sans) !important; font-weight: 600 !important;
     font-size: 1.1rem !important; letter-spacing: 0.06em !important;
     text-transform: uppercase !important; color: var(--muted) !important;
     margin-bottom: 24px !important; }
h3 { font-family: var(--sans) !important; font-weight: 600 !important;
     color: #ffffff !important; font-size: 0.9rem !important;
     letter-spacing: 0.08em !important; text-transform: uppercase !important; }

/* Divider */
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 24px 0 !important; }

/* Metric tile */
.tile {
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 20px 24px;
    margin-bottom: 12px;
}
.tile-val {
    font-family: var(--mono);
    font-size: 2rem;
    font-weight: 500;
    color: #ffffff;
    line-height: 1;
}
.tile-lbl {
    font-size: 11px;
    font-weight: 500;
    color: var(--muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 6px;
}
.tile-accent { border-left: 2px solid var(--accent); }

/* Match row */
.match-row {
    border: 1px solid var(--border);
    border-left: 2px solid var(--border2);
    background: var(--surface);
    padding: 16px 20px;
    margin-bottom: 8px;
    transition: border-color 0.15s;
}
.match-row:hover { border-color: var(--border2); border-left-color: var(--accent); }
.match-label {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 6px;
}
.match-teams {
    font-size: 15px;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 12px;
}
.match-vs { color: var(--muted); font-weight: 400; margin: 0 8px; }

/* Probability bar */
.prob-row { display: flex; align-items: center; margin-bottom: 5px; }
.prob-label {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
    width: 100px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.prob-track {
    flex: 1;
    background: var(--surface2);
    height: 3px;
    margin: 0 12px;
}
.prob-fill-home { background: var(--home); height: 3px; }
.prob-fill-draw { background: var(--draw); height: 3px; }
.prob-fill-away { background: var(--away); height: 3px; }
.prob-num {
    font-family: var(--mono);
    font-size: 12px;
    font-weight: 500;
    color: var(--text);
    width: 42px;
    text-align: right;
}

/* Prediction tag */
.tag {
    display: inline-block;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 3px 8px;
    margin-top: 8px;
    font-family: var(--mono);
}
.tag-home { background: rgba(79,159,255,0.12); color: var(--home); border: 1px solid rgba(79,159,255,0.25); }
.tag-draw { background: rgba(255,204,0,0.12);  color: var(--draw); border: 1px solid rgba(255,204,0,0.25); }
.tag-away { background: rgba(255,68,102,0.12); color: var(--away); border: 1px solid rgba(255,68,102,0.25); }

/* Section label */
.section-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 20px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
}

/* Groups grid */
.group-tile {
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 14px 16px;
    margin-bottom: 8px;
}
.group-letter {
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 500;
    color: var(--accent);
    letter-spacing: 0.1em;
    margin-bottom: 6px;
}
.group-team {
    font-size: 13px;
    font-weight: 500;
    color: var(--text);
    padding: 2px 0;
}

/* Big number display */
.big-prob {
    font-family: var(--mono);
    font-size: 3rem;
    font-weight: 500;
    line-height: 1;
}
.big-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 8px;
}
.prob-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 28px 24px;
    text-align: center;
}
.prob-panel.active-home { border-top: 2px solid var(--home); }
.prob-panel.active-draw { border-top: 2px solid var(--draw); }
.prob-panel.active-away { border-top: 2px solid var(--away); }

/* Result row */
.result-row {
    display: flex;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid var(--border);
    font-size: 13px;
}
.result-correct { color: var(--home); font-family: var(--mono); font-size: 11px; }
.result-wrong   { color: var(--away); font-family: var(--mono); font-size: 11px; }

/* Selectbox */
.stSelectbox > div > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 0 !important;
    font-family: var(--sans) !important;
}

/* Button */
.stButton > button {
    background: var(--accent) !important;
    color: #000000 !important;
    border: none !important;
    border-radius: 0 !important;
    font-family: var(--mono) !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 12px 32px !important;
    width: 100% !important;
}
.stButton > button:hover { background: #6fb0ff !important; }

/* Date input */
.stDateInput > div > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 !important;
}

/* Radio */
.stRadio > div { gap: 0 !important; }

/* Hide streamlit branding */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* Hide the sidebar collapse arrow */
button[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebar"] > div:first-child { padding-top: 2rem !important; }

/* Nav buttons — make them invisible as buttons, look like nav items */
section[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: none !important;
    color: transparent !important;
    font-size: 1px !important;
    padding: 0 !important;
    height: 1px !important;
    margin: -6px 0 4px 0 !important;
    width: 100% !important;
    cursor: pointer !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f: model = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f: le = pickle.load(f)
    return model, le

@st.cache_data
def load_upcoming():
    return pd.read_csv('upcoming_2026.csv', parse_dates=['date'])

@st.cache_data
def load_past():
    if os.path.exists('predictions.csv'):
        return pd.read_csv('predictions.csv', parse_dates=['date'])
    return None

@st.cache_data
def load_rankings():
    df = pd.read_csv('fifa_ranking-2023-07-20.csv', parse_dates=['rank_date'])
    return df[['rank', 'country_full', 'rank_date']].sort_values('rank_date')

@st.cache_data
def load_results_data():
    return pd.read_csv(
        'https://raw.githubusercontent.com/martj42/international_results/master/results.csv',
        parse_dates=['date']
    )

try:
    model, le = load_model()
    upcoming_df = load_upcoming()
    past_df = load_past()
    ranking_df = load_rankings()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    st.error(f"Error loading model: {e}")

# ── Helpers ───────────────────────────────────────────────────────────────────
NAME_MAP = {
    'United States': 'USA', 'South Korea': 'Korea Republic',
    'Iran': 'IR Iran', 'Bosnia and Herzegovina': 'Bosnia-Herzegovina',
    'Ivory Coast': "Côte d'Ivoire", 'Cape Verde': 'Cape Verde Islands',
    'DR Congo': 'DR Congo', 'Czech Republic': 'Czechia',
    'Czechia': 'Czechia', 'Turkey': 'Türkiye', 'Curacao': 'Curaçao',
}

TEAM_ALIASES = {
    # Name changes adopted by martj42 dataset
    'Czech Republic':         ['Czech Republic', 'Czechia'],
    'Czechia':                ['Czechia', 'Czech Republic'],
    'Turkey':                 ['Turkey', 'Türkiye'],
    'Türkiye':                ['Türkiye', 'Turkey'],
    # African team names
    'Ivory Coast':            ['Ivory Coast', "Côte d'Ivoire"],
    "Côte d'Ivoire":         ["Côte d'Ivoire", 'Ivory Coast'],
    'DR Congo':               ['DR Congo', 'Congo DR', 'Democratic Republic of the Congo'],
    'Cape Verde':             ['Cape Verde', 'Cape Verde Islands'],
    # Special characters
    'Curacao':                ['Curacao', 'Curaçao'],
    'Curaçao':                ['Curaçao', 'Curacao'],
    # Long names
    'Bosnia and Herzegovina': ['Bosnia and Herzegovina', 'Bosnia-Herzegovina', 'Bosnia & Herzegovina'],
    'North Macedonia':        ['North Macedonia', 'Macedonia'],
    # US name variants
    'United States':          ['United States', 'USA', 'United States of America'],
    # South Korea
    'South Korea':            ['South Korea', 'Korea Republic', 'Korea, Republic of'],
    # Iran
    'Iran':                   ['Iran', 'IR Iran'],
    # Saudi Arabia
    'Saudi Arabia':           ['Saudi Arabia', 'KSA'],
}

def resolve_names(team):
    return TEAM_ALIASES.get(team, [team])

FEATURE_COLS = ['h_form','h_gf','h_ga','h_gd','h_h2h','h_rank',
                'a_form','a_gf','a_ga','a_gd','a_h2h','a_rank',
                'rank_diff','form_diff','gf_diff','gd_diff','neutral']

def get_rank(team, match_date, default=80):
    mapped = NAME_MAP.get(team, team)
    td = ranking_df[ranking_df['country_full'] == mapped]
    if len(td) == 0: return default
    past = td[td['rank_date'] <= match_date]
    return int(past.iloc[-1]['rank']) if len(past) > 0 else int(td.iloc[0]['rank'])

def get_team_form(team, match_date, opponent, all_results, window_months=12):
    cutoff = match_date - pd.DateOffset(months=window_months)
    names = resolve_names(team)
    mask = (
        (all_results['date'] < match_date) & (all_results['date'] >= cutoff) &
        (all_results['home_team'].isin(names) | all_results['away_team'].isin(names))
    )
    tm = all_results[mask]
    if len(tm) == 0:
        return {'form_score': 0.5, 'goals_scored': 1.0, 'goals_conceded': 1.0,
                'goal_diff': 0.0, 'h2h_winrate': 0.5}
    tm = tm.copy()
    tm['is_friendly'] = tm['tournament'].str.contains('Friendly', case=False, na=False)
    form_scores, gf_list, ga_list = [], [], []
    for _, row in tm.iterrows():
        is_home = row['home_team'] in names
        gf = row['home_score'] if is_home else row['away_score']
        ga = row['away_score'] if is_home else row['home_score']
        opp = row['away_team'] if is_home else row['home_team']
        opp_rank = get_rank(opp, row['date'])
        w = (1 / np.sqrt(opp_rank)) * (0.5 if row.get('is_friendly', False) else 1.0)
        form_scores.append(w * (1 if gf > ga else 0))
        gf_list.append(gf); ga_list.append(ga)
    opp_names = resolve_names(opponent)
    h2h = all_results[
        (all_results['date'] < match_date) &
        (
            (all_results['home_team'].isin(names)     & all_results['away_team'].isin(opp_names)) |
            (all_results['away_team'].isin(names) & all_results['home_team'].isin(opp_names))
        )
    ]
    h2h_rate = 0.5
    if len(h2h) > 0:
        wins = sum(1 for _, r in h2h.iterrows()
                   if (r['home_team'] == team and r['home_score'] > r['away_score']) or
                      (r['away_team'] == team and r['away_score'] > r['home_score']))
        h2h_rate = wins / len(h2h)
    gs, gc = np.mean(gf_list), np.mean(ga_list)
    return {'form_score': np.mean(form_scores), 'goals_scored': gs,
            'goals_conceded': gc, 'goal_diff': gs - gc, 'h2h_winrate': h2h_rate}

def predict_match(home, away, match_date):
    rd = load_results_data()
    rd['is_friendly'] = rd['tournament'].str.contains('Friendly', case=False, na=False)
    hf = get_team_form(home, match_date, away, rd)
    af = get_team_form(away, match_date, home, rd)
    hr, ar = get_rank(home, match_date), get_rank(away, match_date)
    X = pd.DataFrame([{
        'h_form': hf['form_score'], 'h_gf': hf['goals_scored'],
        'h_ga': hf['goals_conceded'], 'h_gd': hf['goal_diff'],
        'h_h2h': hf['h2h_winrate'], 'h_rank': hr,
        'a_form': af['form_score'], 'a_gf': af['goals_scored'],
        'a_ga': af['goals_conceded'], 'a_gd': af['goal_diff'],
        'a_h2h': af['h2h_winrate'], 'a_rank': ar,
        'rank_diff': hr - ar, 'form_diff': hf['form_score'] - af['form_score'],
        'gf_diff': hf['goals_scored'] - af['goals_scored'],
        'gd_diff': hf['goal_diff'] - af['goal_diff'],
        'neutral': 1,
    }])
    proba = model.predict_proba(X)[0]
    classes = list(le.classes_)
    return {
        'home_win': round(proba[classes.index('home_win')] * 100, 1),
        'draw':     round(proba[classes.index('draw')]     * 100, 1),
        'away_win': round(proba[classes.index('away_win')] * 100, 1),
        'pred':     classes[np.argmax(proba)],
    }

def prob_bar_html(label, pct, cls):
    return f"""
    <div class="prob-row">
        <div class="prob-label">{label[:14]}</div>
        <div class="prob-track"><div class="prob-fill-{cls}" style="width:{pct}%;"></div></div>
        <div class="prob-num">{pct:05.2f}%</div>
    </div>"""

def tag_html(pred, home, away):
    if 'home' in pred:
        return f'<span class="tag tag-home">{home[:12]}</span>'
    elif 'draw' in pred:
        return '<span class="tag tag-draw">Draw</span>'
    else:
        return f'<span class="tag tag-away">{away[:12]}</span>'

# ── Navigation state ─────────────────────────────────────────────────────────
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-size:18px;font-weight:700;letter-spacing:-0.01em;color:#fff;margin-bottom:4px;">World Cup Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:11px;color:#52527a;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:20px;">2026 FIFA World Cup</div>', unsafe_allow_html=True)
    st.markdown("---")
    for item in ["Home", "Schedule", "Predictor", "Results"]:
        is_active = st.session_state.page == item
        color  = "#4f9fff" if is_active else "#d0d0e8"
        border = "border-left:2px solid #4f9fff;padding-left:10px;" if is_active else "border-left:2px solid #1c1c30;padding-left:10px;"
        st.markdown(f'<div style="font-size:12px;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;padding:5px 0;{border}color:{color};">{item}</div>', unsafe_allow_html=True)
        if st.button(item, key=f"nav_{item}", use_container_width=True):
            st.session_state.page = item
            st.rerun()
    st.markdown("---")
    st.markdown('<div style="font-size:11px;color:#52527a;line-height:1.8;letter-spacing:0.04em;">MODEL — XGBoost<br>TRAINED — 2010–2022<br>ACCURACY — 54.7%<br>MATCHES — 192 tested</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div style="font-size:11px;color:#52527a;">github.com/sooro-kim</div>', unsafe_allow_html=True)

page = st.session_state.page

if not MODEL_LOADED:
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "Home":
    st.markdown("# World Cup Predictor")
    st.markdown("## Machine learning match outcome predictions")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    tiles = [
        ("54.7%", "Model Accuracy"),
        ("42.7%", "Naive Baseline"),
        ("192", "Matches Validated"),
        ("72", "2026 Predictions"),
    ]
    for col, (val, lbl) in zip([c1, c2, c3, c4], tiles):
        with col:
            st.markdown(f'<div class="tile tile-accent"><div class="tile-val">{val}</div><div class="tile-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("""
<div style="margin-bottom:40px;">
    <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#52527a;letter-spacing:0.18em;text-transform:uppercase;margin-bottom:32px;">
        2026 · FIFA World Cup · North America
    </div>
    <div style="font-family:'Space Grotesk',sans-serif;font-size:3.8rem;font-weight:700;line-height:0.95;letter-spacing:-0.03em;color:#ffffff;margin-bottom:24px;">
        Who<br>wins<br><span style="color:#4f9fff;">tonight.</span>
    </div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:12px;color:#52527a;letter-spacing:0.06em;line-height:2;border-left:1px solid #1c1c30;padding-left:16px;margin-top:28px;">
        XGBoost · Walk-forward validation<br>
        48 teams · 104 matches · June 11 – July 19
    </div>
</div>
""", unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="section-label">2026 Groups</div>', unsafe_allow_html=True)
        GROUPS = {
            'A': ['Mexico', 'South Korea', 'South Africa', 'Czech Republic'],
            'B': ['Canada', 'Switzerland', 'Qatar', 'Bosnia-Herzegovina'],
            'C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'],
            'D': ['United States', 'Australia', 'Paraguay', 'Turkey'],
            'E': ['Germany', 'Ecuador', 'Ivory Coast', 'Curacao'],
            'F': ['Netherlands', 'Japan', 'Sweden', 'Tunisia'],
            'G': ['Belgium', 'Iran', 'Egypt', 'New Zealand'],
            'H': ['Spain', 'Uruguay', 'Saudi Arabia', 'Cape Verde'],
            'I': ['France', 'Senegal', 'Norway', 'Iraq'],
            'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
            'K': ['Portugal', 'Colombia', 'DR Congo', 'Uzbekistan'],
            'L': ['England', 'Croatia', 'Ghana', 'Panama'],
        }
        for g, teams in GROUPS.items():
            teams_html = ''.join(f'<div class="group-team">{t}</div>' for t in teams)
            st.markdown(f'<div class="group-tile"><div class="group-letter">GROUP {g}</div>{teams_html}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SCHEDULE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Schedule":
    st.markdown("# 2026 Group Stage")
    st.markdown("## All 72 matches — predicted probabilities")
    st.markdown("---")

    if upcoming_df is None:
        st.error("upcoming_2026.csv not found.")
        st.stop()

    col1, col2 = st.columns([2, 3])
    with col1:
        all_groups = ['All'] + sorted(upcoming_df['group'].unique().tolist())
        selected = st.selectbox("Group", all_groups)
    with col2:
        today = pd.Timestamp('today').normalize()
        view = st.radio("Filter", ["All", "Upcoming", "Played"], horizontal=True)

    filtered = upcoming_df.copy()
    if selected != 'All':
        filtered = filtered[filtered['group'] == selected]
    if view == "Upcoming":
        filtered = filtered[filtered['date'] >= today]
    elif view == "Played":
        filtered = filtered[filtered['date'] < today]

    st.markdown(f'<div style="font-size:11px;color:#52527a;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:16px;">{len(filtered)} matches</div>', unsafe_allow_html=True)

    for grp in sorted(filtered['group'].unique()):
        st.markdown(f'<div style="font-size:11px;font-weight:600;letter-spacing:0.14em;text-transform:uppercase;color:#4f9fff;padding:12px 0 8px 0;border-bottom:1px solid #1c1c30;margin-bottom:8px;">{grp}</div>', unsafe_allow_html=True)
        grp_df = filtered[filtered['group'] == grp].sort_values('date')
        for _, row in grp_df.iterrows():
            d = row['date'].strftime('%d %b') if hasattr(row['date'], 'strftime') else str(row['date'])[:10]
            hw, dr, aw = row['home_win_pct'], row['draw_pct'], row['away_win_pct']
            pred = row['prediction'].lower()
            rank_line = f"#{int(row['home_rank'])} · {int(row['away_rank'])}#"
            html = f"""
            <div class="match-row">
                <div class="match-label">{d} &nbsp;·&nbsp; {rank_line}</div>
                <div class="match-teams">{row['home_team']}<span class="match-vs">—</span>{row['away_team']}</div>
                {prob_bar_html(row['home_team'], hw, 'home')}
                {prob_bar_html('Draw', dr, 'draw')}
                {prob_bar_html(row['away_team'], aw, 'away')}
                {tag_html(pred, row['home_team'], row['away_team'])}
            </div>"""
            st.markdown(html, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Predictor":
    st.markdown("# Match Predictor")
    st.markdown("## Select two teams to generate a probability forecast")
    st.markdown("---")

    ALL_TEAMS = sorted([
        'Algeria','Argentina','Australia','Austria','Belgium','Bolivia',
        'Bosnia and Herzegovina','Brazil','Cameroon','Canada','Cape Verde',
        'Chile','Colombia','Costa Rica','Croatia','Czech Republic','Denmark',
        'DR Congo','Ecuador','Egypt','England','France','Germany','Ghana',
        'Greece','Haiti','Honduras','Iran','Iraq','Ireland','Italy',
        'Ivory Coast','Jamaica','Japan','Jordan','Mexico','Morocco',
        'Netherlands','New Zealand','Nigeria','Norway','Panama','Paraguay',
        'Peru','Poland','Portugal','Qatar','Romania','Russia','Saudi Arabia',
        'Scotland','Senegal','Serbia','Slovakia','Slovenia','South Africa',
        'South Korea','Spain','Sweden','Switzerland','Tunisia','Turkey',
        'United States','Uruguay','Uzbekistan','Wales',
    ])

    c1, c2, c3 = st.columns([5, 1, 5])
    with c1:
        st.markdown('<div style="font-size:11px;color:#52527a;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:6px;">Home Team</div>', unsafe_allow_html=True)
        home_team = st.selectbox("home", ALL_TEAMS, index=ALL_TEAMS.index('Brazil'), label_visibility="collapsed")
    with c2:
        st.markdown('<div style="text-align:center;padding-top:28px;font-family:\'IBM Plex Mono\',monospace;font-size:13px;color:#52527a;">vs</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div style="font-size:11px;color:#52527a;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:6px;">Away Team</div>', unsafe_allow_html=True)
        away_opts = [t for t in ALL_TEAMS if t != home_team]
        away_team = st.selectbox("away", away_opts, index=away_opts.index('Argentina') if 'Argentina' in away_opts else 0, label_visibility="collapsed")

    st.markdown('<div style="font-size:11px;color:#52527a;letter-spacing:0.1em;text-transform:uppercase;margin:16px 0 6px 0;">Match Date</div>', unsafe_allow_html=True)
    match_date = st.date_input("date", value=date(2026, 6, 15), label_visibility="collapsed")
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Run Prediction"):
        with st.spinner("Computing..."):
            try:
                p = predict_match(home_team, away_team, pd.Timestamp(match_date))
                pred = p['pred']

                st.markdown("---")
                st.markdown(f'<div style="font-size:11px;color:#52527a;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:6px;">{match_date.strftime("%d %b %Y")}</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="font-size:1.4rem;font-weight:700;color:#fff;margin-bottom:20px;">{home_team} — {away_team}</div>', unsafe_allow_html=True)

                home_cls = "active-home" if pred == 'home_win' else ""
                draw_cls = "active-draw" if pred == 'draw' else ""
                away_cls = "active-away" if pred == 'away_win' else ""

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f'<div class="prob-panel {home_cls}"><div class="big-prob" style="color:{"#4f9fff" if pred=="home_win" else "#fff"};">{p["home_win"]:05.2f}%</div><div class="big-label">{home_team}</div></div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="prob-panel {draw_cls}"><div class="big-prob" style="color:{"#ffcc00" if pred=="draw" else "#fff"};">{p["draw"]:05.2f}%</div><div class="big-label">Draw</div></div>', unsafe_allow_html=True)
                with c3:
                    st.markdown(f'<div class="prob-panel {away_cls}"><div class="big-prob" style="color:{"#ff4466" if pred=="away_win" else "#fff"};">{p["away_win"]:05.2f}%</div><div class="big-label">{away_team}</div></div>', unsafe_allow_html=True)

                pred_text = home_team if pred == 'home_win' else ('Draw' if pred == 'draw' else away_team)
                st.markdown(f'<div style="margin-top:16px;font-family:\'IBM Plex Mono\',monospace;font-size:12px;color:#52527a;letter-spacing:0.08em;">PREDICTION — <span style="color:#fff;">{pred_text.upper()}</span></div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Results":
    st.markdown("# Validation Results")
    st.markdown("## Walk-forward testing across 2014, 2018, and 2022")
    st.markdown("---")

    if past_df is None:
        st.warning("predictions.csv not found.")
        st.stop()

    total   = len(past_df)
    correct = past_df['correct'].sum()
    acc     = correct / total * 100
    baseline = past_df['actual'].value_counts(normalize=True).max() * 100
    beat    = acc > baseline

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="tile tile-accent"><div class="tile-val">{acc:.1f}%</div><div class="tile-lbl">Accuracy</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="tile"><div class="tile-val">{baseline:.1f}%</div><div class="tile-lbl">Baseline</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="tile"><div class="tile-val">{correct}/{total}</div><div class="tile-lbl">Correct</div></div>', unsafe_allow_html=True)
    with c4:
        val = "YES" if beat else "NO"
        color = "#4f9fff" if beat else "#ff4466"
        st.markdown(f'<div class="tile"><div class="tile-val" style="color:{color};">{val}</div><div class="tile-lbl">Beat Baseline</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-label">Per Tournament</div>', unsafe_allow_html=True)

    yc = st.columns(3)
    for i, yr in enumerate([2014, 2018, 2022]):
        yd = past_df[past_df['year'] == yr]
        ya = yd['correct'].mean() * 100
        with yc[i]:
            st.markdown(f'<div class="tile"><div class="tile-val">{ya:.1f}%</div><div class="tile-lbl">{yr} — {yd["correct"].sum()}/{len(yd)}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns([2, 3])
    with col1:
        yr_f = st.selectbox("Year", ['All'] + [str(y) for y in sorted(past_df['year'].unique())])
    with col2:
        show_f = st.radio("Show", ['All', 'Correct', 'Incorrect'], horizontal=True)

    disp = past_df.copy()
    if yr_f != 'All': disp = disp[disp['year'] == int(yr_f)]
    if show_f == 'Correct':   disp = disp[disp['correct'] == True]
    if show_f == 'Incorrect': disp = disp[disp['correct'] == False]
    disp = disp.sort_values('date', ascending=False)

    st.markdown(f'<div style="font-size:11px;color:#52527a;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:16px;">{len(disp)} matches</div>', unsafe_allow_html=True)

    # Header
    st.markdown('<div style="display:flex;padding:8px 0;border-bottom:1px solid #1c1c30;font-size:11px;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:#52527a;"><span style="flex:2;">Match</span><span style="flex:1;">Actual</span><span style="flex:1;">Predicted</span><span style="flex:1;text-align:right;">Result</span></div>', unsafe_allow_html=True)

    for _, row in disp.iterrows():
        d = row['date'].strftime('%d %b %Y') if hasattr(row['date'], 'strftime') else str(row['date'])[:10]
        ok = row['correct']
        result_html = f'<span class="result-{"correct" if ok else "wrong"}">{"CORRECT" if ok else "INCORRECT"}</span>'
        actual = row['actual'].replace('_', ' ').title()
        pred   = row['predicted'].replace('_', ' ').title()
        st.markdown(f"""
        <div style="display:flex;align-items:center;padding:10px 0;border-bottom:1px solid #1c1c30;font-size:13px;">
            <span style="flex:2;"><span style="color:#fff;font-weight:500;">{row['home_team']} — {row['away_team']}</span><br><span style="font-size:11px;color:#52527a;">{d} · {row['year']}</span></span>
            <span style="flex:1;color:#d0d0e8;">{actual}</span>
            <span style="flex:1;color:#d0d0e8;">{pred}</span>
            <span style="flex:1;text-align:right;">{result_html}</span>
        </div>""", unsafe_allow_html=True)

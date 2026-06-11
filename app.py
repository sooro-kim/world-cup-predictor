import streamlit as st
import pandas as pd
import numpy as np
import pickle
import urllib.request
import json
from datetime import date, datetime
import os

st.set_page_config(
    page_title="World Cup Predictor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root {
    --bg:      #07070f;
    --surface: #0d0d1c;
    --surface2:#121224;
    --border:  #1c1c30;
    --border2: #2a2a40;
    --text:    #d0d0e8;
    --muted:   #52527a;
    --accent:  #4f9fff;
    --home:    #4f9fff;
    --away:    #ff4466;
    --win:     #00cc88;
    --loss:    #ff4466;
    --mono:    'IBM Plex Mono', monospace;
    --sans:    'Space Grotesk', sans-serif;
}
html, body, [class*="css"] {
    font-family: var(--sans) !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
.stApp { background-color: var(--bg) !important; }
section[data-testid="stSidebar"] { display: none !important; }
button[data-testid="collapsedControl"] { display: none !important; }
.block-container { padding: 1.5rem 2.5rem !important; max-width: 1400px !important; }
h1 { font-family:var(--sans)!important; font-weight:700!important; font-size:1.8rem!important;
     letter-spacing:-0.02em!important; color:#ffffff!important; margin-bottom:2px!important; }
h2 { font-family:var(--sans)!important; font-weight:600!important; font-size:0.85rem!important;
     letter-spacing:0.1em!important; text-transform:uppercase!important;
     color:var(--muted)!important; margin-bottom:20px!important; }
h3 { font-family:var(--sans)!important; font-weight:600!important; color:#fff!important;
     font-size:0.85rem!important; letter-spacing:0.08em!important;
     text-transform:uppercase!important; }
hr  { border:none!important; border-top:1px solid var(--border)!important; margin:20px 0!important; }

/* Tiles */
.tile { background:var(--surface); border:1px solid var(--border);
        border-left:2px solid var(--accent); padding:16px 20px; margin-bottom:10px; }
.tile-val { font-family:var(--mono); font-size:1.8rem; font-weight:500;
            color:#fff; line-height:1; }
.tile-lbl { font-size:10px; font-weight:600; color:var(--muted);
            letter-spacing:0.14em; text-transform:uppercase; margin-top:5px; }

/* Nav buttons */
.stButton > button {
    background: transparent !important; border: 1px solid var(--border) !important;
    color: var(--muted) !important; font-family: var(--mono) !important;
    font-size: 11px !important; font-weight: 600 !important;
    letter-spacing: 0.12em !important; text-transform: uppercase !important;
    padding: 8px 0 !important; border-radius: 0 !important; width: 100% !important;
}
.stButton > button:hover {
    background: var(--surface2) !important; color: #fff !important;
    border-color: var(--accent) !important;
}

/* Match card */
.match-card { background:var(--surface); border:1px solid var(--border);
              border-left:2px solid var(--border2); padding:14px 18px;
              margin-bottom:8px; }
.match-card:hover { border-left-color:var(--accent); }
.match-meta { font-family:var(--mono); font-size:10px; color:var(--muted);
              letter-spacing:0.08em; margin-bottom:5px; }
.match-teams { font-size:14px; font-weight:600; color:#fff; margin-bottom:10px; }
.match-vs { color:var(--muted); font-weight:400; margin:0 8px; }

/* Prob bars */
.prob-row { display:flex; align-items:center; margin-bottom:4px; }
.prob-label { font-family:var(--mono); font-size:10px; color:var(--muted);
              width:100px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.prob-track { flex:1; background:var(--surface2); height:3px; margin:0 10px; }
.prob-fill-home { background:var(--home); height:3px; }
.prob-fill-away { background:var(--away); height:3px; }
.prob-num { font-family:var(--mono); font-size:11px; font-weight:500;
            color:var(--text); width:48px; text-align:right; }

/* Tags */
.tag { display:inline-block; font-family:var(--mono); font-size:10px;
       font-weight:600; letter-spacing:0.1em; text-transform:uppercase;
       padding:2px 8px; margin-top:6px; }
.tag-home { background:rgba(79,159,255,0.1); color:var(--home);
            border:1px solid rgba(79,159,255,0.2); }
.tag-away { background:rgba(255,68,102,0.1); color:var(--away);
            border:1px solid rgba(255,68,102,0.2); }
.tag-correct  { background:rgba(0,204,136,0.1); color:var(--win);
                border:1px solid rgba(0,204,136,0.2); }
.tag-wrong    { background:rgba(255,68,102,0.1); color:var(--loss);
                border:1px solid rgba(255,68,102,0.2); }
.tag-live     { background:rgba(255,204,0,0.1); color:#ffcc00;
                border:1px solid rgba(255,204,0,0.25); }

/* Group header */
.grp-hdr { font-family:var(--mono); font-size:10px; font-weight:600;
           letter-spacing:0.16em; text-transform:uppercase; color:var(--accent);
           padding:10px 0 6px 0; border-bottom:1px solid var(--border);
           margin-bottom:8px; }

/* Predictor panels */
.prob-panel { background:var(--surface); border:1px solid var(--border);
              padding:24px 20px; text-align:center; }
.prob-panel.active-home { border-top:2px solid var(--home); }
.prob-panel.active-away { border-top:2px solid var(--away); }
.big-prob { font-family:var(--mono); font-size:2.8rem; font-weight:500; line-height:1; }
.big-lbl  { font-size:10px; font-weight:600; letter-spacing:0.12em;
            text-transform:uppercase; color:var(--muted); margin-top:8px; }

/* Selectbox */
.stSelectbox > div > div {
    background:var(--surface)!important; border:1px solid var(--border)!important;
    color:var(--text)!important; border-radius:0!important; font-family:var(--sans)!important;
}
/* Date input */
.stDateInput > div > div {
    background:var(--surface)!important; border:1px solid var(--border)!important;
    border-radius:0!important;
}
/* Group tiles */
.group-tile { background:var(--surface); border:1px solid var(--border);
              padding:12px 14px; margin-bottom:6px; }
.group-letter { font-family:var(--mono); font-size:10px; font-weight:600;
                color:var(--accent); letter-spacing:0.1em; margin-bottom:5px; }
.group-team { font-size:12px; font-weight:500; color:var(--text); padding:1px 0; }

/* Results table */
.result-row { display:flex; align-items:center; padding:8px 0;
              border-bottom:1px solid var(--border); font-size:12px; }

#MainMenu { visibility:hidden; } footer { visibility:hidden; } header { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

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

@st.cache_data(ttl=86400)  # refresh once per day
def load_live_results():
    """Fetch actual match results from openfootball — updated daily."""
    try:
        url = "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json"
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read())
        matches = data.get('matches', [])
        results = {}
        for m in matches:
            if m.get('score'):
                ft = m['score'].get('ft', [None, None])
                if ft[0] is not None and ft[1] is not None:
                    # Normalize team names
                    t1 = normalize_name(m['team1'])
                    t2 = normalize_name(m['team2'])
                    results[(t1, t2)] = {
                        'home_score': int(ft[0]),
                        'away_score': int(ft[1]),
                        'date': m['date'],
                    }
        return results
    except Exception:
        return {}

def normalize_name(name):
    """Map openfootball names to our team names."""
    mapping = {
        'Bosnia & Herzegovina': 'Bosnia and Herzegovina',
        'USA':                  'United States',
        'Korea Republic':       'South Korea',
        'Czechia':              'Czech Republic',
        'Türkiye':              'Turkey',
        "Côte d'Ivoire":        'Ivory Coast',
        'Cape Verde Islands':   'Cape Verde',
        'Congo DR':             'DR Congo',
        'IR Iran':              'Iran',
    }
    return mapping.get(name, name)

try:
    model     = load_model()
    upcoming  = load_upcoming()
    past_df   = load_past()
    ranking_df = load_rankings()
    live      = load_live_results()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    st.error(f"Error loading model: {e}")

# ── Helpers ───────────────────────────────────────────────────────────────────
TEAM_ALIASES = {
    'Czech Republic':         ['Czech Republic', 'Czechia'],
    'Czechia':                ['Czechia', 'Czech Republic'],
    'Turkey':                 ['Turkey', 'Türkiye'],
    'Türkiye':                ['Türkiye', 'Turkey'],
    'Ivory Coast':            ['Ivory Coast', "Côte d'Ivoire"],
    "Côte d'Ivoire":         ["Côte d'Ivoire", 'Ivory Coast'],
    'DR Congo':               ['DR Congo', 'Congo DR'],
    'Cape Verde':             ['Cape Verde', 'Cape Verde Islands'],
    'Curacao':                ['Curacao', 'Curaçao'],
    'Curaçao':                ['Curaçao', 'Curacao'],
    'Bosnia and Herzegovina': ['Bosnia and Herzegovina', 'Bosnia-Herzegovina',
                               'Bosnia & Herzegovina'],
    'United States':          ['United States', 'USA'],
    'South Korea':            ['South Korea', 'Korea Republic'],
    'Iran':                   ['Iran', 'IR Iran'],
}

NAME_MAP = {
    'United States': 'USA', 'South Korea': 'Korea Republic',
    'Iran': 'IR Iran', 'Bosnia and Herzegovina': 'Bosnia-Herzegovina',
    'Ivory Coast': "Côte d'Ivoire", 'Cape Verde': 'Cape Verde Islands',
    'DR Congo': 'DR Congo', 'Czech Republic': 'Czechia',
    'Czechia': 'Czechia', 'Turkey': 'Türkiye', 'Curacao': 'Curaçao',
}

FEATURE_COLS = [
    'h_form','h_gf','h_ga','h_gd','h_h2h','h_rank',
    'a_form','a_gf','a_ga','a_gd','a_h2h','a_rank',
    'rank_diff','form_diff','gf_diff','gd_diff','neutral'
]

def resolve_names(team):
    return TEAM_ALIASES.get(team, [team])

def get_rank(team, match_date, default=80):
    mapped = NAME_MAP.get(team, team)
    td = ranking_df[ranking_df['country_full'] == mapped]
    if len(td) == 0: return default
    past = td[td['rank_date'] <= match_date]
    return int(past.iloc[-1]['rank']) if len(past) > 0 else int(td.iloc[0]['rank'])

@st.cache_data(ttl=3600)
def load_results_data():
    return pd.read_csv(
        'https://raw.githubusercontent.com/martj42/international_results/master/results.csv',
        parse_dates=['date']
    )

def get_team_form(team, match_date, opponent, all_results, window_months=12):
    cutoff = match_date - pd.DateOffset(months=window_months)
    names  = resolve_names(team)
    mask   = (
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
                   if (r['home_team'] in names and r['home_score'] > r['away_score']) or
                      (r['away_team'] in names and r['away_score'] > r['home_score']))
        h2h_rate = wins / len(h2h)
    gs, gc = np.mean(gf_list), np.mean(ga_list)
    return {'form_score': np.mean(form_scores), 'goals_scored': gs,
            'goals_conceded': gc, 'goal_diff': gs - gc, 'h2h_winrate': h2h_rate}

def predict_match(home, away, match_date):
    rd = load_results_data()
    hf = get_team_form(home, match_date, away, rd)
    af = get_team_form(away, match_date, home, rd)
    hr = get_rank(home, match_date)
    ar = get_rank(away, match_date)
    X = pd.DataFrame([{
        'h_form': hf['form_score'], 'h_gf': hf['goals_scored'],
        'h_ga':   hf['goals_conceded'], 'h_gd': hf['goal_diff'],
        'h_h2h':  hf['h2h_winrate'],  'h_rank': hr,
        'a_form': af['form_score'], 'a_gf': af['goals_scored'],
        'a_ga':   af['goals_conceded'], 'a_gd': af['goal_diff'],
        'a_h2h':  af['h2h_winrate'],  'a_rank': ar,
        'rank_diff': hr - ar,
        'form_diff': hf['form_score'] - af['form_score'],
        'gf_diff':   hf['goals_scored'] - af['goals_scored'],
        'gd_diff':   hf['goal_diff'] - af['goal_diff'],
        'neutral': 1,
    }])
    prob = float(np.clip(model.predict(X)[0], 0, 1))
    return {
        'home_win': round(prob * 100, 1),
        'away_win': round((1 - prob) * 100, 1),
        'pred':     'home_win' if prob >= 0.5 else 'away_win',
    }

def prob_bar(label, pct, side):
    return (f'<div class="prob-row">'
            f'<div class="prob-label">{label[:14]}</div>'
            f'<div class="prob-track"><div class="prob-fill-{side}" style="width:{pct}%;"></div></div>'
            f'<div class="prob-num">{pct:.1f}%</div>'
            f'</div>')

def pred_tag(pred, home, away):
    if pred == 'home_win' or pred == 'Home Win':
        return f'<span class="tag tag-home">{home[:12]}</span>'
    return f'<span class="tag tag-away">{away[:12]}</span>'

# ── Navigation ────────────────────────────────────────────────────────────────
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

c0, c1, c2, c3, c4 = st.columns([3, 1, 1, 1, 1])
with c0:
    st.markdown('<div style="font-size:15px;font-weight:700;color:#fff;padding-top:10px;'
                'letter-spacing:-0.01em;">World Cup Predictor</div>', unsafe_allow_html=True)
with c1:
    if st.button("Home",      use_container_width=True): st.session_state.page="Home";      st.rerun()
with c2:
    if st.button("Schedule",  use_container_width=True): st.session_state.page="Schedule";  st.rerun()
with c3:
    if st.button("Predictor", use_container_width=True): st.session_state.page="Predictor"; st.rerun()
with c4:
    if st.button("Results",   use_container_width=True): st.session_state.page="Results";   st.rerun()

st.markdown("---")
page = st.session_state.page

if not MODEL_LOADED:
    st.stop()

n_live = len(live)

# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == 'Home':
    st.markdown("# World Cup Predictor")
    st.markdown("## Machine learning match outcome predictions")

    c1, c2, c3, c4 = st.columns(4)
    for col, (val, lbl) in zip([c1,c2,c3,c4], [
        ("67.5%",     "Model Accuracy"),
        ("192",       "Matches Validated"),
        ("72",        "2026 Predictions"),
        (str(n_live), "Results Tracked"),
    ]):
        with col:
            st.markdown(f'<div class="tile"><div class="tile-val">{val}</div>'
                        f'<div class="tile-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("""
<div style="margin-bottom:32px;">
    <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#52527a;
                letter-spacing:0.18em;text-transform:uppercase;margin-bottom:28px;">
        2026 · FIFA World Cup · North America
    </div>
    <div style="font-family:'Space Grotesk',sans-serif;font-size:3.6rem;font-weight:700;
                line-height:0.95;letter-spacing:-0.03em;color:#ffffff;margin-bottom:20px;">
        Who<br>wins<br><span style="color:#4f9fff;">tonight.</span>
    </div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#52527a;
                letter-spacing:0.06em;line-height:2;border-left:1px solid #1c1c30;padding-left:14px;">
        XGBoost Regressor · Walk-forward validation<br>
        Fractional draw targets · Win / Loss predictions<br>
        Live results via openfootball · Updated daily<br>
        48 teams · 104 matches · June 11 – July 19
    </div>
</div>
""", unsafe_allow_html=True)

    with col_r:
        st.markdown('<div style="font-size:10px;font-weight:600;letter-spacing:0.14em;'
                    'text-transform:uppercase;color:#4f9fff;margin-bottom:12px;'
                    'padding-bottom:8px;border-bottom:1px solid #1c1c30;">2026 Groups</div>',
                    unsafe_allow_html=True)
        GROUPS = {
            'A':['Mexico','South Korea','South Africa','Czech Republic'],
            'B':['Canada','Switzerland','Qatar','Bosnia and Herzegovina'],
            'C':['Brazil','Morocco','Haiti','Scotland'],
            'D':['United States','Australia','Paraguay','Turkey'],
            'E':['Germany','Ecuador','Ivory Coast','Curacao'],
            'F':['Netherlands','Japan','Sweden','Tunisia'],
            'G':['Belgium','Iran','Egypt','New Zealand'],
            'H':['Spain','Uruguay','Saudi Arabia','Cape Verde'],
            'I':['France','Senegal','Norway','Iraq'],
            'J':['Argentina','Algeria','Austria','Jordan'],
            'K':['Portugal','Colombia','DR Congo','Uzbekistan'],
            'L':['England','Croatia','Ghana','Panama'],
        }
        for g, teams in GROUPS.items():
            teams_html = ''.join(f'<div class="group-team">{t}</div>' for t in teams)
            st.markdown(f'<div class="group-tile"><div class="group-letter">GROUP {g}</div>'
                        f'{teams_html}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SCHEDULE
# ══════════════════════════════════════════════════════════════════════════════
elif page == 'Schedule':
    st.markdown("# 2026 Group Stage")
    st.markdown("## Predicted win probabilities · Results auto-update daily")

    col1, col2 = st.columns([2, 3])
    with col1:
        all_groups = ['All'] + sorted(upcoming['group'].unique().tolist())
        sel = st.selectbox("Group", all_groups)
    with col2:
        today = pd.Timestamp('today').normalize()
        view  = st.radio("Show", ["All","Upcoming","Played"], horizontal=True)

    filtered = upcoming.copy()
    if sel != 'All': filtered = filtered[filtered['group'] == sel]
    if view == 'Upcoming': filtered = filtered[filtered['date'] >= today]
    elif view == 'Played': filtered = filtered[filtered['date'] < today]

    n_results = len(live)
    st.markdown(f'<div style="font-size:10px;color:#52527a;letter-spacing:0.1em;'
                f'text-transform:uppercase;margin-bottom:14px;">'
                f'{len(filtered)} matches · {n_results} results loaded</div>',
                unsafe_allow_html=True)

    for grp in sorted(filtered['group'].unique()):
        st.markdown(f'<div class="grp-hdr">{grp}</div>', unsafe_allow_html=True)
        grp_df = filtered[filtered['group'] == grp].sort_values('date')
        for _, row in grp_df.iterrows():
            d    = row['date'].strftime('%d %b') if hasattr(row['date'],'strftime') else str(row['date'])[:10]
            hw   = row['home_win_pct']
            aw   = row['away_win_pct']
            pred = row['prediction']
            home = row['home_team']
            away = row['away_team']

            # Check if live result exists
            result = live.get((home, away), live.get((away, home), None))
            result_html = ''
            correct_tag = ''
            if result:
                hs, as_ = result['home_score'], result['away_score']
                if result.get('home_team') == away:
                    hs, as_ = as_, hs
                actual = 'home_win' if hs > as_ else ('away_win' if as_ > hs else 'draw')
                pred_lower = pred.lower().replace(' ', '_')
                score_str  = f"{hs} – {as_}"
                result_html = (f'&nbsp;&nbsp;<span style="font-family:var(--mono);'
                               f'font-size:10px;color:#ffcc00;">{score_str}</span>')
                if actual != 'draw':
                    ok = (actual == pred_lower)
                    correct_tag = (f'<span class="tag tag-correct">Correct</span>'
                                   if ok else f'<span class="tag tag-wrong">Wrong</span>')

            st.markdown(
                f'<div class="match-card">'
                f'<div class="match-meta">{d}{result_html}</div>'
                f'<div class="match-teams">{home}<span class="match-vs">—</span>{away}</div>'
                f'{prob_bar(home, hw, "home")}'
                f'{prob_bar(away, aw, "away")}'
                f'<div style="margin-top:6px;">{pred_tag(pred, home, away)}'
                f'{"&nbsp;" + correct_tag if correct_tag else ""}</div>'
                f'</div>', unsafe_allow_html=True
            )

# ══════════════════════════════════════════════════════════════════════════════
# PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == 'Predictor':
    st.markdown("# Match Predictor")
    st.markdown("## Select two teams to generate a win probability forecast")

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
        st.markdown('<div style="font-size:10px;color:#52527a;letter-spacing:0.1em;'
                    'text-transform:uppercase;margin-bottom:6px;">Home Team</div>',
                    unsafe_allow_html=True)
        home_team = st.selectbox("home", ALL_TEAMS,
                                 index=ALL_TEAMS.index('Brazil'),
                                 label_visibility="collapsed")
    with c2:
        st.markdown('<div style="text-align:center;padding-top:28px;font-family:'
                    '\'IBM Plex Mono\',monospace;font-size:12px;color:#52527a;">vs</div>',
                    unsafe_allow_html=True)
    with c3:
        st.markdown('<div style="font-size:10px;color:#52527a;letter-spacing:0.1em;'
                    'text-transform:uppercase;margin-bottom:6px;">Away Team</div>',
                    unsafe_allow_html=True)
        away_opts = [t for t in ALL_TEAMS if t != home_team]
        away_team = st.selectbox("away", away_opts,
                                 index=away_opts.index('Argentina') if 'Argentina' in away_opts else 0,
                                 label_visibility="collapsed")

    st.markdown('<div style="font-size:10px;color:#52527a;letter-spacing:0.1em;'
                'text-transform:uppercase;margin:14px 0 6px 0;">Match Date</div>',
                unsafe_allow_html=True)
    match_date = st.date_input("date", value=date(2026, 6, 15),
                               min_value=date(2011, 1, 1),
                               max_value=date(2027, 12, 31),
                               label_visibility="collapsed")
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Run Prediction"):
        too_far = pd.Timestamp(match_date) > pd.Timestamp.today() + pd.DateOffset(months=3)
        if too_far:
            st.warning("Not enough recent data for a date this far ahead. "
                       "Select a date within the next 3 months.")
        else:
            with st.spinner("Computing..."):
                try:
                    p    = predict_match(home_team, away_team, pd.Timestamp(match_date))
                    pred = p['pred']
                    st.markdown("---")
                    st.markdown(
                        f'<div style="font-size:10px;color:#52527a;letter-spacing:0.1em;'
                        f'text-transform:uppercase;margin-bottom:5px;">'
                        f'{match_date.strftime("%d %b %Y")}</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div style="font-size:1.3rem;font-weight:700;color:#fff;'
                        f'margin-bottom:18px;">{home_team} — {away_team}</div>',
                        unsafe_allow_html=True)

                    c1, c2 = st.columns(2)
                    with c1:
                        cls = "active-home" if pred == 'home_win' else ""
                        col = "#4f9fff" if pred == 'home_win' else "#fff"
                        st.markdown(
                            f'<div class="prob-panel {cls}">'
                            f'<div class="big-prob" style="color:{col};">{p["home_win"]:05.1f}%</div>'
                            f'<div class="big-lbl">{home_team}</div>'
                            f'</div>', unsafe_allow_html=True)
                    with c2:
                        cls = "active-away" if pred == 'away_win' else ""
                        col = "#ff4466" if pred == 'away_win' else "#fff"
                        st.markdown(
                            f'<div class="prob-panel {cls}">'
                            f'<div class="big-prob" style="color:{col};">{p["away_win"]:05.1f}%</div>'
                            f'<div class="big-lbl">{away_team}</div>'
                            f'</div>', unsafe_allow_html=True)

                    winner = home_team if pred == 'home_win' else away_team
                    st.markdown(
                        f'<div style="margin-top:14px;font-family:\'IBM Plex Mono\',monospace;'
                        f'font-size:11px;color:#52527a;letter-spacing:0.08em;">'
                        f'PREDICTION — <span style="color:#fff;">{winner.upper()}</span>'
                        f'</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == 'Results':
    st.markdown("# Validation Results")
    st.markdown("## Walk-forward testing · 2014, 2018, 2022")

    # Live accuracy from 2026 tournament
    live_correct = 0
    live_total   = 0
    for (home, away), res in live.items():
        match_row = upcoming[(upcoming['home_team']==home) & (upcoming['away_team']==away)]
        if len(match_row) == 0: continue
        hs, as_ = res['home_score'], res['away_score']
        if hs == as_: continue  # skip draws for accuracy
        actual = 'Home Win' if hs > as_ else 'Away Win'
        pred   = match_row.iloc[0]['prediction']
        live_total   += 1
        if actual == pred: live_correct += 1

    c1, c2, c3, c4 = st.columns(4)
    if past_df is not None:
        decisive = past_df[past_df['correct'].notna()]
        total   = len(decisive)
        correct = decisive['correct'].sum()
        acc     = correct / total * 100 if total > 0 else 0
    else:
        total, correct, acc = 0, 0, 0

    for col, (val, lbl) in zip([c1,c2,c3,c4], [
        (f"{acc:.1f}%",        "Historical Accuracy"),
        (f"{correct}/{total}", "Correct (2014–2022)"),
        (f"{live_correct}/{live_total}" if live_total > 0 else "—",
         "2026 Live Accuracy"),
        (str(n_live),          "2026 Results In"),
    ]):
        with col:
            st.markdown(f'<div class="tile"><div class="tile-val">{val}</div>'
                        f'<div class="tile-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    if live_total > 0:
        st.markdown('<div style="font-size:10px;font-weight:600;letter-spacing:0.14em;'
                    'text-transform:uppercase;color:#4f9fff;margin-bottom:12px;'
                    'padding-bottom:8px;border-bottom:1px solid #1c1c30;">2026 Live Results</div>',
                    unsafe_allow_html=True)
        st.markdown('<div style="display:flex;padding:6px 0;border-bottom:1px solid #1c1c30;'
                    'font-size:10px;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;'
                    'color:#52527a;"><span style="flex:3">Match</span>'
                    '<span style="flex:1">Score</span>'
                    '<span style="flex:1">Predicted</span>'
                    '<span style="flex:1;text-align:right">Result</span></div>',
                    unsafe_allow_html=True)
        for (home, away), res in sorted(live.items(), key=lambda x: x[1]['date']):
            match_row = upcoming[(upcoming['home_team']==home) & (upcoming['away_team']==away)]
            if len(match_row) == 0: continue
            hs, as_ = res['home_score'], res['away_score']
            pred = match_row.iloc[0]['prediction']
            if hs == as_:
                result_tag = '<span class="tag tag-live">Draw</span>'
            else:
                actual = 'Home Win' if hs > as_ else 'Away Win'
                ok = (actual == pred)
                result_tag = (f'<span class="tag tag-correct">Correct</span>'
                              if ok else f'<span class="tag tag-wrong">Wrong</span>')
            st.markdown(
                f'<div style="display:flex;align-items:center;padding:8px 0;'
                f'border-bottom:1px solid #1c1c30;font-size:12px;">'
                f'<span style="flex:3;color:#fff;font-weight:500;">{home} — {away}</span>'
                f'<span style="flex:1;font-family:var(--mono);color:#d0d0e8;">{hs}–{as_}</span>'
                f'<span style="flex:1;color:#d0d0e8;">{pred}</span>'
                f'<span style="flex:1;text-align:right;">{result_tag}</span>'
                f'</div>', unsafe_allow_html=True)
        st.markdown("---")

    if past_df is not None:
        st.markdown('<div style="font-size:10px;font-weight:600;letter-spacing:0.14em;'
                    'text-transform:uppercase;color:#4f9fff;margin-bottom:12px;'
                    'padding-bottom:8px;border-bottom:1px solid #1c1c30;">'
                    'Historical Validation (2014–2022)</div>', unsafe_allow_html=True)

        yr_cols = st.columns(3)
        for i, yr in enumerate([2014, 2018, 2022]):
            yd = past_df[past_df['year'] == yr]
            yd_dec = yd[yd['correct'].notna()]
            ya = yd_dec['correct'].mean() * 100 if len(yd_dec) > 0 else 0
            with yr_cols[i]:
                st.markdown(
                    f'<div class="tile"><div class="tile-val">{ya:.1f}%</div>'
                    f'<div class="tile-lbl">{yr} — '
                    f'{int(yd_dec["correct"].sum())}/{len(yd_dec)}</div></div>',
                    unsafe_allow_html=True)

        st.markdown("---")
        col1, col2 = st.columns([2, 3])
        with col1:
            yr_f = st.selectbox("Year", ['All'] + [str(y) for y in sorted(past_df['year'].unique())])
        with col2:
            show_f = st.radio("Show", ['All','Correct','Incorrect'], horizontal=True)

        disp = past_df.copy()
        if yr_f != 'All': disp = disp[disp['year'] == int(yr_f)]
        if show_f == 'Correct':   disp = disp[disp['correct'] == True]
        if show_f == 'Incorrect': disp = disp[disp['correct'] == False]
        disp = disp[disp['correct'].notna()].sort_values('date', ascending=False)

        st.markdown(f'<div style="font-size:10px;color:#52527a;letter-spacing:0.08em;'
                    f'text-transform:uppercase;margin-bottom:12px;">'
                    f'{len(disp)} matches</div>', unsafe_allow_html=True)

        st.markdown('<div style="display:flex;padding:6px 0;border-bottom:1px solid #1c1c30;'
                    'font-size:10px;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;'
                    'color:#52527a;"><span style="flex:3">Match</span>'
                    '<span style="flex:1">Actual</span>'
                    '<span style="flex:1">Predicted</span>'
                    '<span style="flex:1;text-align:right">Result</span></div>',
                    unsafe_allow_html=True)

        for _, row in disp.iterrows():
            d   = row['date'].strftime('%d %b %Y') if hasattr(row['date'],'strftime') else str(row['date'])[:10]
            ok  = row['correct']
            tag = ('<span class="tag tag-correct">Correct</span>'
                   if ok else '<span class="tag tag-wrong">Wrong</span>')
            st.markdown(
                f'<div style="display:flex;align-items:center;padding:8px 0;'
                f'border-bottom:1px solid #1c1c30;font-size:12px;">'
                f'<span style="flex:3;color:#fff;font-weight:500;">{row["home_team"]} — '
                f'{row["away_team"]}<br>'
                f'<span style="font-size:10px;color:#52527a;">{d} · {row["year"]}</span></span>'
                f'<span style="flex:1;color:#d0d0e8;">'
                f'{row["actual"].replace("_"," ").title()}</span>'
                f'<span style="flex:1;color:#d0d0e8;">'
                f'{row["predicted"].replace("_"," ").title()}</span>'
                f'<span style="flex:1;text-align:right;">{tag}</span>'
                f'</div>', unsafe_allow_html=True)

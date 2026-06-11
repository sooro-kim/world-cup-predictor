import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import date
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="World Cup Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1a5c38;
    }
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 16px !important;
        font-weight: 600 !important;
    }

    /* Main */
    .main { background-color: #f4faf6; }

    /* Headings */
    h1 { color: #1a5c38 !important; font-weight: 900 !important; font-size: 2.4rem !important; }
    h2 { color: #1a5c38 !important; font-weight: 700 !important; }
    h3 { color: #1a5c38 !important; font-weight: 600 !important; }

    /* Metric card */
    .metric-card {
        background: white;
        border-top: 4px solid #1a5c38;
        border-radius: 10px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 900;
        color: #1a5c38;
        line-height: 1;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #666;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Match card */
    .match-card {
        background: white;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.07);
        border-left: 5px solid #1a5c38;
    }
    .match-teams {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1a1a1a;
    }
    .match-date {
        font-size: 0.8rem;
        color: #888;
        margin-bottom: 8px;
    }
    .prob-row {
        display: flex;
        align-items: center;
        margin-bottom: 4px;
        font-size: 0.85rem;
    }
    .prob-label { width: 90px; color: #555; }
    .prob-bar-bg {
        flex: 1;
        background: #e8f5ee;
        border-radius: 4px;
        height: 14px;
        margin: 0 8px;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        background: #1a5c38;
        border-radius: 4px;
    }
    .prob-val { width: 40px; text-align: right; font-weight: 700; color: #1a5c38; }

    /* Prediction badge */
    .badge-win  { background:#1a5c38; color:white; padding:3px 10px; border-radius:12px; font-size:0.8rem; font-weight:700; }
    .badge-draw { background:#f0a500; color:white; padding:3px 10px; border-radius:12px; font-size:0.8rem; font-weight:700; }
    .badge-away { background:#c0392b; color:white; padding:3px 10px; border-radius:12px; font-size:0.8rem; font-weight:700; }

    /* Group header */
    .group-header {
        background: #1a5c38;
        color: white;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1rem;
        margin: 16px 0 8px 0;
    }

    /* Correct/Incorrect */
    .correct   { color: #1a5c38; font-weight: 700; }
    .incorrect { color: #c0392b; font-weight: 700; }

    /* Buttons */
    .stButton > button {
        background-color: #1a5c38 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        background-color: #145030 !important;
    }

    /* Divider */
    hr { border-color: #c8e6d4; }
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    return model, le

@st.cache_data
def load_upcoming():
    return pd.read_csv('upcoming_2026.csv', parse_dates=['date'])

@st.cache_data
def load_results():
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
    past_df = load_results()
    ranking_df = load_rankings()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    st.error(f"Error loading model: {e}. Please run save_model.py first.")

# ── Feature helpers ───────────────────────────────────────────────────────────
NAME_MAP = {
    'United States':          'USA',
    'South Korea':            'Korea Republic',
    'Iran':                   'IR Iran',
    'Bosnia and Herzegovina': 'Bosnia-Herzegovina',
    'Ivory Coast':            "Côte d'Ivoire",
    'Cape Verde':             'Cape Verde Islands',
    'DR Congo':               'DR Congo',
    'Czech Republic':         'Czechia',
    'Czechia':                'Czechia',
    'Turkey':                 'Türkiye',
    'Curacao':                'Curaçao',
}

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
    mask = (
        (all_results['date'] < match_date) & (all_results['date'] >= cutoff) &
        ((all_results['home_team'] == team) | (all_results['away_team'] == team))
    )
    tm = all_results[mask]
    if len(tm) == 0:
        return {'form_score': 0.5, 'goals_scored': 1.0, 'goals_conceded': 1.0,
                'goal_diff': 0.0, 'h2h_winrate': 0.5}
    tm['is_friendly'] = tm['tournament'].str.contains('Friendly', case=False, na=False)
    form_scores, gf_list, ga_list = [], [], []
    for _, row in tm.iterrows():
        is_home = row['home_team'] == team
        gf = row['home_score'] if is_home else row['away_score']
        ga = row['away_score'] if is_home else row['home_score']
        opp = row['away_team'] if is_home else row['home_team']
        opp_rank = get_rank(opp, row['date'])
        w = (1 / np.sqrt(opp_rank)) * (0.5 if row.get('is_friendly', False) else 1.0)
        form_scores.append(w * (1 if gf > ga else 0))
        gf_list.append(gf); ga_list.append(ga)
    h2h = all_results[
        (all_results['date'] < match_date) &
        (((all_results['home_team'] == team) & (all_results['away_team'] == opponent)) |
         ((all_results['away_team'] == team) & (all_results['home_team'] == opponent)))
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
    results_data = load_results_data()
    results_data['is_friendly'] = results_data['tournament'].str.contains('Friendly', case=False, na=False)
    hf = get_team_form(home, match_date, away, results_data)
    af = get_team_form(away, match_date, home, results_data)
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

def prob_bar(label, pct, color='#1a5c38'):
    return f"""
    <div class="prob-row">
        <div class="prob-label">{label}</div>
        <div class="prob-bar-bg">
            <div class="prob-bar-fill" style="width:{pct}%;background:{color};"></div>
        </div>
        <div class="prob-val">{pct}%</div>
    </div>"""

def pred_badge(pred):
    if 'home' in pred.lower():
        return '<span class="badge-win">Home Win</span>'
    elif 'draw' in pred.lower():
        return '<span class="badge-draw">Draw</span>'
    else:
        return '<span class="badge-away">Away Win</span>'

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚽ World Cup Predictor")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "📅 Upcoming Matches", "🔮 Custom Prediction", "📊 Past Results"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Model:** XGBoost")
    st.markdown("**Data:** 2010 – 2022 World Cups")
    st.markdown("**Features:** Form, Goals, H2H, FIFA Rank")
    st.markdown("---")
    st.markdown("Made by [sooro-kim](https://github.com/sooro-kim)")

if not MODEL_LOADED:
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("# ⚽ World Cup Predictor")
    st.markdown("**Machine learning predictions for every 2026 FIFA World Cup match.**")
    st.markdown("---")

    # Stats row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-value">54.7%</div><div class="metric-label">Model Accuracy</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-value">42.7%</div><div class="metric-label">Baseline Accuracy</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-value">192</div><div class="metric-label">Matches Tested</div></div>', unsafe_allow_html=True)
    with col4:
        n = len(upcoming_df) if upcoming_df is not None else 72
        st.markdown(f'<div class="metric-card"><div class="metric-value">{n}</div><div class="metric-label">2026 Predictions</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.markdown("### How It Works")
        st.markdown("""
The model is an **XGBoost classifier** trained on all four World Cups from 2010 to 2022.
For each match, it computes rolling features from the 12 months of international football
played by each team before the match.

**Features used:**
- **Quality-adjusted form score** — win rate weighted by opponent FIFA ranking. Beating Brazil counts more than beating Curacao. Friendly wins count at half weight.
- **Goals scored / conceded** — average per match over the past 12 months
- **Head-to-head win rate** — historical record between the two specific teams
- **FIFA ranking difference** — pre-match ranking gap between the two teams

**Validation method:** Walk-forward — train on all tournaments before the test year,
predict that year without ever seeing future data.
        """)

        st.markdown("### Tournament Info")
        st.markdown("""
| | |
|---|---|
| **Dates** | June 11 – July 19, 2026 |
| **Host nations** | USA, Canada, Mexico |
| **Teams** | 48 (expanded from 32) |
| **Total matches** | 104 |
| **Groups** | 12 groups of 4 |
| **Format** | Top 2 + 8 best 3rd-place → Round of 32 |
        """)

    with col_right:
        st.markdown("### 2026 Groups")
        groups = {
            'A': 'Mexico · South Korea · South Africa · Czech Republic',
            'B': 'Canada · Switzerland · Qatar · Bosnia-Herzegovina',
            'C': 'Brazil · Morocco · Haiti · Scotland',
            'D': 'USA · Australia · Paraguay · Turkey',
            'E': 'Germany · Ecuador · Ivory Coast · Curaçao',
            'F': 'Netherlands · Japan · Sweden · Tunisia',
            'G': 'Belgium · Iran · Egypt · New Zealand',
            'H': 'Spain · Uruguay · Saudi Arabia · Cape Verde',
            'I': 'France · Senegal · Norway · Iraq',
            'J': 'Argentina · Algeria · Austria · Jordan',
            'K': 'Portugal · Colombia · DR Congo · Uzbekistan',
            'L': 'England · Croatia · Ghana · Panama',
        }
        for g, teams in groups.items():
            st.markdown(f"**Group {g}:** {teams}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — UPCOMING MATCHES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📅 Upcoming Matches":
    st.markdown("# 📅 2026 World Cup — Group Stage Predictions")
    st.markdown("All 72 group stage matches predicted using the XGBoost model.")
    st.markdown("---")

    if upcoming_df is None:
        st.error("upcoming_2026.csv not found. Please run save_model.py first.")
        st.stop()

    # Filter controls
    col1, col2 = st.columns([2, 3])
    with col1:
        all_groups = ['All Groups'] + sorted(upcoming_df['group'].unique().tolist())
        selected_group = st.selectbox("Filter by Group", all_groups)
    with col2:
        today = pd.Timestamp('today').normalize()
        view = st.radio("Show", ["All Matches", "Upcoming Only", "Played Only"],
                        horizontal=True)

    filtered = upcoming_df.copy()
    if selected_group != 'All Groups':
        filtered = filtered[filtered['group'] == selected_group]
    if view == "Upcoming Only":
        filtered = filtered[filtered['date'] >= today]
    elif view == "Played Only":
        filtered = filtered[filtered['date'] < today]

    st.markdown(f"**Showing {len(filtered)} matches**")

    # Display by group
    for grp in sorted(filtered['group'].unique()):
        st.markdown(f'<div class="group-header">{grp}</div>', unsafe_allow_html=True)
        grp_matches = filtered[filtered['group'] == grp].sort_values('date')

        for _, row in grp_matches.iterrows():
            date_str = row['date'].strftime('%b %d, %Y') if hasattr(row['date'], 'strftime') else str(row['date'])[:10]
            hw = row['home_win_pct']
            dr = row['draw_pct']
            aw = row['away_win_pct']
            pred = row['prediction']

            html = f"""
            <div class="match-card">
                <div class="match-date">{date_str}</div>
                <div class="match-teams">{row['home_team']} <span style="color:#888;font-weight:400">vs</span> {row['away_team']}</div>
                <div style="margin:8px 0 4px 0;font-size:0.8rem;color:#888;">
                    FIFA Rank: {row['home_team']} #{int(row['home_rank'])} &nbsp;|&nbsp; {row['away_team']} #{int(row['away_rank'])}
                </div>
                {prob_bar(row['home_team'][:12], hw)}
                {prob_bar('Draw', dr, '#f0a500')}
                {prob_bar(row['away_team'][:12], aw, '#c0392b')}
                <div style="margin-top:8px;">Prediction: {pred_badge(pred)}</div>
            </div>
            """
            st.markdown(html, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — CUSTOM PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Custom Prediction":
    st.markdown("# 🔮 Custom Match Prediction")
    st.markdown("Pick any two teams and get a prediction.")
    st.markdown("---")

    ALL_TEAMS = sorted([
        'Argentina', 'Australia', 'Algeria', 'Austria', 'Belgium', 'Bolivia',
        'Bosnia and Herzegovina', 'Brazil', 'Cameroon', 'Canada', 'Cape Verde',
        'Chile', 'Colombia', 'Costa Rica', 'Croatia', 'Cuba', 'Czech Republic',
        'Denmark', 'DR Congo', 'Ecuador', 'Egypt', 'England', 'France',
        'Germany', 'Ghana', 'Greece', 'Haiti', 'Honduras', 'Hungary', 'Iran',
        'Iraq', 'Ireland', 'Italy', 'Ivory Coast', 'Jamaica', 'Japan', 'Jordan',
        'Mexico', 'Morocco', 'Netherlands', 'New Zealand', 'Nigeria', 'Norway',
        'Panama', 'Paraguay', 'Peru', 'Poland', 'Portugal', 'Qatar', 'Romania',
        'Russia', 'Saudi Arabia', 'Scotland', 'Senegal', 'Serbia', 'Slovakia',
        'Slovenia', 'South Africa', 'South Korea', 'Spain', 'Sweden',
        'Switzerland', 'Tunisia', 'Turkey', 'United States', 'Uruguay',
        'Uzbekistan', 'Wales',
    ])

    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        home_team = st.selectbox("Home Team", ALL_TEAMS, index=ALL_TEAMS.index('Brazil'))
    with col2:
        st.markdown("<div style='text-align:center;padding-top:35px;font-size:1.5rem;font-weight:900;color:#1a5c38;'>VS</div>", unsafe_allow_html=True)
    with col3:
        away_options = [t for t in ALL_TEAMS if t != home_team]
        away_team = st.selectbox("Away Team", away_options, index=away_options.index('Argentina') if 'Argentina' in away_options else 0)

    match_date = st.date_input("Match Date", value=date(2026, 6, 15))
    st.markdown("")

    if st.button("⚽ Predict Match"):
        with st.spinner("Computing prediction..."):
            try:
                pred = predict_match(home_team, away_team, pd.Timestamp(match_date))

                st.markdown("---")
                st.markdown(f"### {home_team} vs {away_team}")
                st.markdown(f"*{match_date.strftime('%B %d, %Y')}*")
                st.markdown("")

                col1, col2, col3 = st.columns(3)
                with col1:
                    color = '#1a5c38' if pred['pred'] == 'home_win' else '#888'
                    st.markdown(f"""
                    <div class="metric-card" style="border-top-color:{color};">
                        <div class="metric-value" style="color:{color};">{pred['home_win']}%</div>
                        <div class="metric-label">{home_team} Win</div>
                    </div>""", unsafe_allow_html=True)
                with col2:
                    color = '#f0a500' if pred['pred'] == 'draw' else '#888'
                    st.markdown(f"""
                    <div class="metric-card" style="border-top-color:{color};">
                        <div class="metric-value" style="color:{color};">{pred['draw']}%</div>
                        <div class="metric-label">Draw</div>
                    </div>""", unsafe_allow_html=True)
                with col3:
                    color = '#c0392b' if pred['pred'] == 'away_win' else '#888'
                    st.markdown(f"""
                    <div class="metric-card" style="border-top-color:{color};">
                        <div class="metric-value" style="color:{color};">{pred['away_win']}%</div>
                        <div class="metric-label">{away_team} Win</div>
                    </div>""", unsafe_allow_html=True)

                pred_label = pred['pred'].replace('_', ' ').title()
                if pred['pred'] == 'home_win':
                    pred_label = f"{home_team} Win"
                elif pred['pred'] == 'away_win':
                    pred_label = f"{away_team} Win"

                st.markdown("")
                st.success(f"**Prediction: {pred_label}**")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PAST RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Past Results":
    st.markdown("# 📊 Past Prediction Results")
    st.markdown("Walk-forward validation across 2014, 2018, and 2022 World Cups.")
    st.markdown("---")

    if past_df is None:
        st.warning("predictions.csv not found. Please run the notebook and save_model.py first.")
        st.stop()

    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    total = len(past_df)
    correct = past_df['correct'].sum()
    acc = correct / total * 100

    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{acc:.1f}%</div><div class="metric-label">Overall Accuracy</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{correct}/{total}</div><div class="metric-label">Correct Predictions</div></div>', unsafe_allow_html=True)
    with col3:
        baseline = past_df['actual'].value_counts(normalize=True).max() * 100
        st.markdown(f'<div class="metric-card"><div class="metric-value">{baseline:.1f}%</div><div class="metric-label">Baseline Accuracy</div></div>', unsafe_allow_html=True)
    with col4:
        beat = "✅ YES" if acc > baseline else "❌ NO"
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="font-size:1.4rem;">{beat}</div><div class="metric-label">Beat Baseline</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Per-year breakdown
    st.markdown("### Accuracy by Tournament")
    yr_cols = st.columns(3)
    for i, yr in enumerate([2014, 2018, 2022]):
        yr_df = past_df[past_df['year'] == yr]
        yr_acc = yr_df['correct'].mean() * 100
        yr_correct = yr_df['correct'].sum()
        yr_total = len(yr_df)
        with yr_cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{yr_acc:.1f}%</div>
                <div class="metric-label">{yr} World Cup<br>{yr_correct}/{yr_total} correct</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Filter
    col1, col2 = st.columns([2, 3])
    with col1:
        yr_filter = st.selectbox("Filter by Year", ['All'] + [str(y) for y in sorted(past_df['year'].unique())])
    with col2:
        show_filter = st.radio("Show", ['All', 'Correct only', 'Incorrect only'], horizontal=True)

    display = past_df.copy()
    if yr_filter != 'All':
        display = display[display['year'] == int(yr_filter)]
    if show_filter == 'Correct only':
        display = display[display['correct'] == True]
    elif show_filter == 'Incorrect only':
        display = display[display['correct'] == False]

    display = display.sort_values('date', ascending=False)

    # Table
    st.markdown(f"**{len(display)} matches shown**")
    for _, row in display.iterrows():
        correct_icon = "✅" if row['correct'] else "❌"
        date_str = row['date'].strftime('%b %d, %Y') if hasattr(row['date'], 'strftime') else str(row['date'])[:10]
        hw = row.get('prob_home_win', '-')
        dr = row.get('prob_draw', '-')
        aw = row.get('prob_away_win', '-')

        actual_label = row['actual'].replace('_', ' ').title()
        pred_label = row['predicted'].replace('_', ' ').title()

        cols = st.columns([3, 2, 2, 2, 1])
        with cols[0]:
            st.markdown(f"**{row['home_team']} vs {row['away_team']}**  \n*{date_str} · {row['year']}*")
        with cols[1]:
            st.markdown(f"Actual: **{actual_label}**")
        with cols[2]:
            st.markdown(f"Predicted: **{pred_label}**")
        with cols[3]:
            if hw != '-':
                st.markdown(f"HW:{hw}% D:{dr}% AW:{aw}%")
        with cols[4]:
            st.markdown(f"**{correct_icon}**")
        st.markdown("---")

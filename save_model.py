"""
save_model.py
Run this ONCE from your project folder (where your notebook is) before deploying.
It trains the model and precomputes all 2026 predictions.

Usage:
    python save_model.py
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

os.makedirs('data', exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading match results...")
results = pd.read_csv(
    'https://raw.githubusercontent.com/martj42/international_results/master/results.csv',
    parse_dates=['date']
)
results['is_friendly'] = results['tournament'].str.contains('Friendly', case=False, na=False)
print(f"  {len(results):,} matches loaded.")

print("Loading FIFA rankings...")
for path in ['data/archive/fifa_ranking-2023-07-20.csv', 'data/fifa_ranking-2023-07-20.csv']:
    if os.path.exists(path):
        ranking_df = pd.read_csv(path, parse_dates=['rank_date'])
        break
ranking_df = ranking_df[['rank', 'country_full', 'rank_date']].sort_values('rank_date')
print(f"  {len(ranking_df):,} ranking entries loaded.")

# ── Name mapping ─────────────────────────────────────────────────────────────
name_map = {
    'United States':          'USA',
    'South Korea':            'Korea Republic',
    'North Korea':            'Korea DPR',
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

def get_rank(team, match_date, default=80):
    mapped = name_map.get(team, team)
    td = ranking_df[ranking_df['country_full'] == mapped]
    if len(td) == 0:
        return default
    past = td[td['rank_date'] <= match_date]
    return int(past.iloc[-1]['rank']) if len(past) > 0 else int(td.iloc[0]['rank'])

def get_team_form(team, match_date, opponent, all_results, window_months=12):
    cutoff = match_date - pd.DateOffset(months=window_months)
    mask = (
        (all_results['date'] < match_date) &
        (all_results['date'] >= cutoff) &
        ((all_results['home_team'] == team) | (all_results['away_team'] == team))
    )
    tm = all_results[mask]
    if len(tm) == 0:
        return {'form_score': 0.5, 'goals_scored': 1.0, 'goals_conceded': 1.0,
                'goal_diff': 0.0, 'h2h_winrate': 0.5}

    form_scores, gf_list, ga_list = [], [], []
    for _, row in tm.iterrows():
        is_home = row['home_team'] == team
        gf = row['home_score'] if is_home else row['away_score']
        ga = row['away_score'] if is_home else row['home_score']
        opp = row['away_team'] if is_home else row['home_team']
        opp_rank = get_rank(opp, row['date'])
        w = (1 / np.sqrt(opp_rank)) * (0.5 if row['is_friendly'] else 1.0)
        form_scores.append(w * (1 if gf > ga else 0))
        gf_list.append(gf)
        ga_list.append(ga)

    h2h = all_results[
        (all_results['date'] < match_date) &
        (((all_results['home_team'] == team) & (all_results['away_team'] == opponent)) |
         ((all_results['away_team'] == team) & (all_results['home_team'] == opponent)))
    ]
    h2h_wins = sum(
        1 for _, r in h2h.iterrows()
        if (r['home_team'] == team and r['home_score'] > r['away_score']) or
           (r['away_team'] == team and r['away_score'] > r['home_score'])
    ) if len(h2h) > 0 else 0
    h2h_rate = h2h_wins / len(h2h) if len(h2h) > 0 else 0.5

    gs, gc = np.mean(gf_list), np.mean(ga_list)
    return {'form_score': np.mean(form_scores), 'goals_scored': gs,
            'goals_conceded': gc, 'goal_diff': gs - gc, 'h2h_winrate': h2h_rate}

def build_row(home, away, date, year, result=None):
    hf = get_team_form(home, date, away, results)
    af = get_team_form(away, date, home, results)
    hr, ar = get_rank(home, date), get_rank(away, date)
    row = {
        'date': date, 'year': year, 'home_team': home, 'away_team': away,
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
    }
    if result: row['result'] = result
    return row

FEATURE_COLS = ['h_form','h_gf','h_ga','h_gd','h_h2h','h_rank',
                'a_form','a_gf','a_ga','a_gd','a_h2h','a_rank',
                'rank_diff','form_diff','gf_diff','gd_diff','neutral']

# ── Build training data ───────────────────────────────────────────────────────
print("\nBuilding training features (3-5 minutes)...")
wc = results[
    results['tournament'].str.contains('FIFA World Cup', case=False, na=False) &
    ~results['tournament'].str.contains('qualif', case=False, na=False)
].copy()
wc['year'] = wc['date'].dt.year
wc = wc[wc['year'].isin([2010, 2014, 2018, 2022])].copy()

def get_result(r):
    if r['home_score'] > r['away_score']: return 'home_win'
    elif r['home_score'] < r['away_score']: return 'away_win'
    return 'draw'
wc['result'] = wc.apply(get_result, axis=1)

rows = []
for i, (_, m) in enumerate(wc.iterrows()):
    if i % 20 == 0: print(f"  {i+1}/{len(wc)}...")
    rows.append(build_row(m['home_team'], m['away_team'], m['date'], m['year'], m['result']))

df = pd.DataFrame(rows)
le = LabelEncoder()
df['result_enc'] = le.fit_transform(df['result'])

# ── Train final model ─────────────────────────────────────────────────────────
print("\nTraining final model...")
model = xgb.XGBClassifier(
    n_estimators=200, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    use_label_encoder=False, eval_metric='mlogloss', random_state=42
)
model.fit(df[FEATURE_COLS], df['result_enc'])

with open('data/model.pkl', 'wb') as f: pickle.dump(model, f)
with open('data/label_encoder.pkl', 'wb') as f: pickle.dump(le, f)
print("Model saved to data/model.pkl")

# Copy FIFA rankings to data root for app
import shutil
if os.path.exists('data/archive/fifa_ranking-2023-07-20.csv'):
    shutil.copy('data/archive/fifa_ranking-2023-07-20.csv', 'data/fifa_ranking-2023-07-20.csv')
    print("FIFA rankings copied to data/fifa_ranking-2023-07-20.csv")

# ── 2026 Group Stage Schedule ─────────────────────────────────────────────────
print("\nComputing 2026 predictions (5-10 minutes)...")

SCHEDULE_2026 = [
    # Group A
    ('A','2026-06-11','Mexico','South Africa'),
    ('A','2026-06-11','South Korea','Czech Republic'),
    ('A','2026-06-15','Mexico','South Korea'),
    ('A','2026-06-15','Czech Republic','South Africa'),
    ('A','2026-06-23','South Africa','South Korea'),
    ('A','2026-06-23','Mexico','Czech Republic'),
    # Group B
    ('B','2026-06-11','Canada','Bosnia and Herzegovina'),
    ('B','2026-06-12','Qatar','Switzerland'),
    ('B','2026-06-16','Canada','Qatar'),
    ('B','2026-06-16','Bosnia and Herzegovina','Switzerland'),
    ('B','2026-06-23','Canada','Switzerland'),
    ('B','2026-06-23','Bosnia and Herzegovina','Qatar'),
    # Group C
    ('C','2026-06-12','Brazil','Haiti'),
    ('C','2026-06-12','Morocco','Scotland'),
    ('C','2026-06-16','Brazil','Morocco'),
    ('C','2026-06-17','Haiti','Scotland'),
    ('C','2026-06-24','Brazil','Scotland'),
    ('C','2026-06-24','Morocco','Haiti'),
    # Group D
    ('D','2026-06-12','United States','Paraguay'),
    ('D','2026-06-13','Australia','Turkey'),
    ('D','2026-06-17','United States','Australia'),
    ('D','2026-06-17','Paraguay','Turkey'),
    ('D','2026-06-24','United States','Turkey'),
    ('D','2026-06-24','Australia','Paraguay'),
    # Group E
    ('E','2026-06-13','Germany','Curacao'),
    ('E','2026-06-13','Ivory Coast','Ecuador'),
    ('E','2026-06-17','Germany','Ivory Coast'),
    ('E','2026-06-18','Ecuador','Curacao'),
    ('E','2026-06-25','Germany','Ecuador'),
    ('E','2026-06-25','Ivory Coast','Curacao'),
    # Group F
    ('F','2026-06-14','Netherlands','Sweden'),
    ('F','2026-06-14','Japan','Tunisia'),
    ('F','2026-06-18','Netherlands','Japan'),
    ('F','2026-06-18','Tunisia','Sweden'),
    ('F','2026-06-25','Netherlands','Tunisia'),
    ('F','2026-06-25','Japan','Sweden'),
    # Group G
    ('G','2026-06-14','Belgium','New Zealand'),
    ('G','2026-06-15','Iran','Egypt'),
    ('G','2026-06-18','Belgium','Iran'),
    ('G','2026-06-19','New Zealand','Egypt'),
    ('G','2026-06-26','Belgium','Egypt'),
    ('G','2026-06-26','Iran','New Zealand'),
    # Group H
    ('H','2026-06-15','Spain','Cape Verde'),
    ('H','2026-06-15','Saudi Arabia','Uruguay'),
    ('H','2026-06-19','Spain','Saudi Arabia'),
    ('H','2026-06-19','Cape Verde','Uruguay'),
    ('H','2026-06-26','Spain','Uruguay'),
    ('H','2026-06-26','Saudi Arabia','Cape Verde'),
    # Group I
    ('I','2026-06-16','France','Iraq'),
    ('I','2026-06-16','Norway','Senegal'),
    ('I','2026-06-19','France','Norway'),
    ('I','2026-06-20','Iraq','Senegal'),
    ('I','2026-06-27','France','Senegal'),
    ('I','2026-06-27','Iraq','Norway'),
    # Group J
    ('J','2026-06-16','Argentina','Jordan'),
    ('J','2026-06-17','Algeria','Austria'),
    ('J','2026-06-20','Argentina','Algeria'),
    ('J','2026-06-20','Austria','Jordan'),
    ('J','2026-06-27','Argentina','Austria'),
    ('J','2026-06-27','Algeria','Jordan'),
    # Group K
    ('K','2026-06-17','Portugal','Uzbekistan'),
    ('K','2026-06-17','Colombia','DR Congo'),
    ('K','2026-06-21','Portugal','Colombia'),
    ('K','2026-06-21','Uzbekistan','DR Congo'),
    ('K','2026-06-27','Portugal','DR Congo'),
    ('K','2026-06-27','Colombia','Uzbekistan'),
    # Group L
    ('L','2026-06-17','England','Panama'),
    ('L','2026-06-18','Croatia','Ghana'),
    ('L','2026-06-21','England','Croatia'),
    ('L','2026-06-22','Panama','Ghana'),
    ('L','2026-06-27','England','Ghana'),
    ('L','2026-06-27','Croatia','Panama'),
]

upcoming_rows = []
classes = list(le.classes_)
for i, (grp, date_str, home, away) in enumerate(SCHEDULE_2026):
    if i % 12 == 0: print(f"  Match {i+1}/{len(SCHEDULE_2026)}...")
    date = pd.Timestamp(date_str)
    r = build_row(home, away, date, 2026)
    X = pd.DataFrame([{c: r[c] for c in FEATURE_COLS}])
    proba = model.predict_proba(X)[0]
    pred_idx = np.argmax(proba)
    upcoming_rows.append({
        'group':         f'Group {grp}',
        'date':          date_str,
        'home_team':     home,
        'away_team':     away,
        'home_win_pct':  round(proba[classes.index('home_win')] * 100, 1),
        'draw_pct':      round(proba[classes.index('draw')]     * 100, 1),
        'away_win_pct':  round(proba[classes.index('away_win')] * 100, 1),
        'prediction':    classes[pred_idx].replace('_', ' ').title(),
        'home_rank':     r['h_rank'],
        'away_rank':     r['a_rank'],
    })

upcoming_df = pd.DataFrame(upcoming_rows)
upcoming_df.to_csv('data/upcoming_2026.csv', index=False)
print(f"Saved {len(upcoming_df)} predictions to data/upcoming_2026.csv")

print("\nAll done! Files ready for GitHub:")
for f in ['data/model.pkl','data/label_encoder.pkl',
          'data/upcoming_2026.csv','data/predictions.csv',
          'data/fifa_ranking-2023-07-20.csv']:
    if os.path.exists(f):
        kb = os.path.getsize(f) // 1024
        print(f"  {f}  ({kb} KB)")
    else:
        print(f"  {f}  MISSING")

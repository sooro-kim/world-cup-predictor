"""
save_model_v2.py
- XGBRegressor with fractional draw targets
- Draw score = 0.5 + (away_rank - home_rank) / 400, capped 0.2-0.8
- Brier score + accuracy evaluation
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import xgboost as xgb

os.makedirs('data', exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading match results...")
results = pd.read_csv(
    'https://raw.githubusercontent.com/martj42/international_results/master/results.csv',
    parse_dates=['date']
)
results['is_friendly'] = results['tournament'].str.contains('Friendly', case=False, na=False)
results = results.sort_values('date').reset_index(drop=True)
print(f"  {len(results):,} matches loaded.")

print("Loading FIFA rankings...")
for path in ['data/archive/fifa_ranking-2023-07-20.csv', 'data/fifa_ranking-2023-07-20.csv']:
    if os.path.exists(path):
        ranking_df = pd.read_csv(path, parse_dates=['rank_date'])
        break
ranking_df = ranking_df[['rank', 'country_full', 'rank_date']].sort_values('rank_date')
print(f"  {len(ranking_df):,} ranking entries loaded.")

# ── Team aliases ─────────────────────────────────────────────────────────────
TEAM_ALIASES = {
    'Czech Republic':         ['Czech Republic', 'Czechia'],
    'Czechia':                ['Czechia', 'Czech Republic'],
    'Turkey':                 ['Turkey', 'Türkiye'],
    'Türkiye':                ['Türkiye', 'Turkey'],
    'Ivory Coast':            ['Ivory Coast', "Côte d'Ivoire"],
    "Côte d'Ivoire":         ["Côte d'Ivoire", 'Ivory Coast'],
    'DR Congo':               ['DR Congo', 'Congo DR', 'Democratic Republic of the Congo'],
    'Cape Verde':             ['Cape Verde', 'Cape Verde Islands'],
    'Curacao':                ['Curacao', 'Curaçao'],
    'Curaçao':                ['Curaçao', 'Curacao'],
    'Bosnia and Herzegovina': ['Bosnia and Herzegovina', 'Bosnia-Herzegovina',
                               'Bosnia & Herzegovina'],
    'North Macedonia':        ['North Macedonia', 'Macedonia'],
    'United States':          ['United States', 'USA'],
    'South Korea':            ['South Korea', 'Korea Republic'],
    'Iran':                   ['Iran', 'IR Iran'],
    'Saudi Arabia':           ['Saudi Arabia', 'KSA'],
}

NAME_MAP = {
    'United States': 'USA', 'South Korea': 'Korea Republic',
    'Iran': 'IR Iran', 'Bosnia and Herzegovina': 'Bosnia-Herzegovina',
    'Ivory Coast': "Côte d'Ivoire", 'Cape Verde': 'Cape Verde Islands',
    'DR Congo': 'DR Congo', 'Czech Republic': 'Czechia',
    'Czechia': 'Czechia', 'Turkey': 'Türkiye', 'Curacao': 'Curaçao',
}

def resolve_names(team):
    return TEAM_ALIASES.get(team, [team])

def get_rank(team, match_date, default=80):
    mapped = NAME_MAP.get(team, team)
    td = ranking_df[ranking_df['country_full'] == mapped]
    if len(td) == 0: return default
    past = td[td['rank_date'] <= match_date]
    return int(past.iloc[-1]['rank']) if len(past) > 0 else int(td.iloc[0]['rank'])

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
    form_scores, gf_list, ga_list = [], [], []
    for _, row in tm.iterrows():
        is_home = row['home_team'] in names
        gf = row['home_score'] if is_home else row['away_score']
        ga = row['away_score'] if is_home else row['home_score']
        opp = row['away_team'] if is_home else row['home_team']
        opp_rank = get_rank(opp, row['date'])
        w = (1 / np.sqrt(opp_rank)) * (0.5 if row['is_friendly'] else 1.0)
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

FEATURE_COLS = [
    'h_form','h_gf','h_ga','h_gd','h_h2h','h_rank',
    'a_form','a_gf','a_ga','a_gd','a_h2h','a_rank',
    'rank_diff','form_diff','gf_diff','gd_diff','neutral'
]

def build_features(home, away, date, year=None):
    hf = get_team_form(home, date, away, results)
    af = get_team_form(away, date, home, results)
    hr = get_rank(home, date)
    ar = get_rank(away, date)
    return {
        'date': date, 'year': year or date.year,
        'home_team': home, 'away_team': away,
        'h_form': hf['form_score'],  'h_gf': hf['goals_scored'],
        'h_ga':   hf['goals_conceded'], 'h_gd': hf['goal_diff'],
        'h_h2h':  hf['h2h_winrate'],  'h_rank': hr,
        'a_form': af['form_score'],  'a_gf': af['goals_scored'],
        'a_ga':   af['goals_conceded'], 'a_gd': af['goal_diff'],
        'a_h2h':  af['h2h_winrate'],  'a_rank': ar,
        'rank_diff': hr - ar,
        'form_diff': hf['form_score'] - af['form_score'],
        'gf_diff':   hf['goals_scored'] - af['goals_scored'],
        'gd_diff':   hf['goal_diff'] - af['goal_diff'],
        'neutral': 1,
    }

# ── Build training dataset ────────────────────────────────────────────────────
print("\nBuilding training features (3-5 minutes)...")
wc = results[
    results['tournament'].str.contains('FIFA World Cup', case=False, na=False) &
    ~results['tournament'].str.contains('qualif', case=False, na=False)
].copy()
wc['year'] = wc['date'].dt.year
wc = wc[wc['year'].isin([2010, 2014, 2018, 2022])].copy()
print(f"  {len(wc)} World Cup matches")

rows = []
for i, (_, m) in enumerate(wc.iterrows()):
    if i % 20 == 0: print(f"  {i+1}/{len(wc)}...")
    r = build_features(m['home_team'], m['away_team'], m['date'], m['year'])

    # ── Fractional draw target ─────────────────────────────────────────────
    # Win  → 1.0
    # Loss → 0.0
    # Draw → 0.5 + (away_rank - home_rank) / 400, capped 0.2-0.8
    # If home team drew against a much stronger away team → above 0.5 (credit)
    # If home team drew against a much weaker away team  → below 0.5 (penalty)
    if m['home_score'] > m['away_score']:
        target = 1.0
    elif m['home_score'] < m['away_score']:
        target = 0.0
    else:
        hr = get_rank(m['home_team'], m['date'])
        ar = get_rank(m['away_team'],  m['date'])
        raw = 0.5 + (ar - hr) / 400.0
        target = float(np.clip(raw, 0.2, 0.8))

    r['target'] = target
    rows.append(r)

df = pd.DataFrame(rows)
print(f"\nTarget distribution:")
print(f"  Wins  (1.0): {(df['target']==1.0).sum()}")
print(f"  Losses(0.0): {(df['target']==0.0).sum()}")
print(f"  Draws (0.2-0.8): {((df['target']>0.0)&(df['target']<1.0)).sum()}")
print(f"  Mean target: {df['target'].mean():.3f}")

# ── Walk-forward validation ───────────────────────────────────────────────────
print("\nWalk-forward validation with GridSearchCV...")

param_grid = {
    'n_estimators':     [100, 200],
    'max_depth':        [2, 3, 4],
    'learning_rate':    [0.01, 0.05, 0.1],
    'subsample':        [0.7, 0.9],
    'colsample_bytree': [0.7, 0.9],
}

wc_years    = [2010, 2014, 2018, 2022]
all_preds   = []
all_actuals = []
all_details = []

for test_year in wc_years:
    train_years = [y for y in wc_years if y < test_year]
    if not train_years:
        print(f"  {test_year}: skipping")
        continue

    train_df = df[df['year'].isin(train_years)]
    test_df  = df[df['year'] == test_year]
    X_train, y_train = train_df[FEATURE_COLS], train_df['target']
    X_test,  y_test  = test_df[FEATURE_COLS],  test_df['target']

    print(f"\n  {test_year}: train={len(X_train)}, test={len(X_test)}")

    grid = GridSearchCV(
        xgb.XGBRegressor(random_state=42),
        param_grid, cv=3, scoring='neg_mean_squared_error',
        n_jobs=-1, verbose=0
    )
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    y_pred = np.clip(best.predict(X_test), 0, 1)

    # Brier score
    brier = mean_squared_error(y_test, y_pred)

    # Accuracy: threshold at 0.5
    # Actual: win if target==1.0, loss if target==0.0, skip draws
    decisive = test_df[test_df['target'].isin([0.0, 1.0])].index
    if len(decisive) > 0:
        y_dec_true = (y_test[decisive] == 1.0).astype(int)
        y_dec_pred = (y_pred[test_df.index.get_indexer(decisive)] >= 0.5).astype(int)
        acc = (y_dec_true.values == y_dec_pred).mean()
    else:
        acc = 0.0

    print(f"  Brier: {brier:.4f}  |  Accuracy (decisive): {acc:.1%}")
    print(f"  Best params: {grid.best_params_}")

    all_preds.extend(y_pred.tolist())
    all_actuals.extend(y_test.values.tolist())

    for j, (_, row) in enumerate(test_df.iterrows()):
        pred_prob = float(y_pred[j])
        actual    = float(y_test.values[j])
        # For display: convert to win/loss
        if actual == 1.0:   actual_label = 'home_win'
        elif actual == 0.0: actual_label = 'away_win'
        else:                actual_label = 'draw'
        pred_label = 'home_win' if pred_prob >= 0.5 else 'away_win'
        # Correct only on decisive matches
        if actual in [0.0, 1.0]:
            correct = (pred_label == actual_label)
        else:
            correct = None  # draw — not counted

        all_details.append({
            'year':          test_year,
            'date':          row['date'],
            'home_team':     row['home_team'],
            'away_team':     row['away_team'],
            'actual':        actual_label,
            'predicted':     pred_label,
            'correct':       correct,
            'home_win_prob': round(pred_prob * 100, 1),
            'away_win_prob': round((1 - pred_prob) * 100, 1),
        })

all_preds_arr = np.array(all_preds)
all_actuals_arr = np.array(all_actuals)
overall_brier = mean_squared_error(all_actuals_arr, all_preds_arr)
decisive_mask = (all_actuals_arr == 0.0) | (all_actuals_arr == 1.0)
if decisive_mask.sum() > 0:
    overall_acc = ((all_preds_arr[decisive_mask] >= 0.5) ==
                   (all_actuals_arr[decisive_mask] == 1.0)).mean()
else:
    overall_acc = 0.0

print(f"\n  Overall Brier score: {overall_brier:.4f}  (lower = better)")
print(f"  Overall accuracy (decisive matches): {overall_acc:.1%}")

details_df = pd.DataFrame(all_details)
details_df.to_csv('data/predictions.csv', index=False)

# ── Train final model ─────────────────────────────────────────────────────────
print("\nTraining final model on all 2010-2022 data...")
final_model = xgb.XGBRegressor(
    n_estimators=200, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42
)
final_model.fit(df[FEATURE_COLS], df['target'])

with open('data/model.pkl', 'wb') as f: pickle.dump(final_model, f)
print("  Model saved.")

# Copy FIFA rankings
import shutil
if os.path.exists('data/archive/fifa_ranking-2023-07-20.csv'):
    shutil.copy('data/archive/fifa_ranking-2023-07-20.csv',
                'data/fifa_ranking-2023-07-20.csv')

# ── 2026 predictions ──────────────────────────────────────────────────────────
print("\nComputing 2026 predictions...")

SCHEDULE_2026 = [
    ('A','2026-06-11','Mexico','South Africa'),
    ('A','2026-06-11','South Korea','Czech Republic'),
    ('A','2026-06-15','Mexico','South Korea'),
    ('A','2026-06-15','Czech Republic','South Africa'),
    ('A','2026-06-23','South Africa','South Korea'),
    ('A','2026-06-23','Mexico','Czech Republic'),
    ('B','2026-06-11','Canada','Bosnia and Herzegovina'),
    ('B','2026-06-12','Qatar','Switzerland'),
    ('B','2026-06-16','Canada','Qatar'),
    ('B','2026-06-16','Bosnia and Herzegovina','Switzerland'),
    ('B','2026-06-23','Canada','Switzerland'),
    ('B','2026-06-23','Bosnia and Herzegovina','Qatar'),
    ('C','2026-06-12','Brazil','Haiti'),
    ('C','2026-06-12','Morocco','Scotland'),
    ('C','2026-06-16','Brazil','Morocco'),
    ('C','2026-06-17','Haiti','Scotland'),
    ('C','2026-06-24','Brazil','Scotland'),
    ('C','2026-06-24','Morocco','Haiti'),
    ('D','2026-06-12','United States','Paraguay'),
    ('D','2026-06-13','Australia','Turkey'),
    ('D','2026-06-17','United States','Australia'),
    ('D','2026-06-17','Paraguay','Turkey'),
    ('D','2026-06-24','United States','Turkey'),
    ('D','2026-06-24','Australia','Paraguay'),
    ('E','2026-06-13','Germany','Curacao'),
    ('E','2026-06-13','Ivory Coast','Ecuador'),
    ('E','2026-06-17','Germany','Ivory Coast'),
    ('E','2026-06-18','Ecuador','Curacao'),
    ('E','2026-06-25','Germany','Ecuador'),
    ('E','2026-06-25','Ivory Coast','Curacao'),
    ('F','2026-06-14','Netherlands','Sweden'),
    ('F','2026-06-14','Japan','Tunisia'),
    ('F','2026-06-18','Netherlands','Japan'),
    ('F','2026-06-18','Tunisia','Sweden'),
    ('F','2026-06-25','Netherlands','Tunisia'),
    ('F','2026-06-25','Japan','Sweden'),
    ('G','2026-06-14','Belgium','New Zealand'),
    ('G','2026-06-15','Iran','Egypt'),
    ('G','2026-06-18','Belgium','Iran'),
    ('G','2026-06-19','New Zealand','Egypt'),
    ('G','2026-06-26','Belgium','Egypt'),
    ('G','2026-06-26','Iran','New Zealand'),
    ('H','2026-06-15','Spain','Cape Verde'),
    ('H','2026-06-15','Saudi Arabia','Uruguay'),
    ('H','2026-06-19','Spain','Saudi Arabia'),
    ('H','2026-06-19','Cape Verde','Uruguay'),
    ('H','2026-06-26','Spain','Uruguay'),
    ('H','2026-06-26','Saudi Arabia','Cape Verde'),
    ('I','2026-06-16','France','Iraq'),
    ('I','2026-06-16','Norway','Senegal'),
    ('I','2026-06-19','France','Norway'),
    ('I','2026-06-20','Iraq','Senegal'),
    ('I','2026-06-27','France','Senegal'),
    ('I','2026-06-27','Iraq','Norway'),
    ('J','2026-06-16','Argentina','Jordan'),
    ('J','2026-06-17','Algeria','Austria'),
    ('J','2026-06-20','Argentina','Algeria'),
    ('J','2026-06-20','Austria','Jordan'),
    ('J','2026-06-27','Argentina','Austria'),
    ('J','2026-06-27','Algeria','Jordan'),
    ('K','2026-06-17','Portugal','Uzbekistan'),
    ('K','2026-06-17','Colombia','DR Congo'),
    ('K','2026-06-21','Portugal','Colombia'),
    ('K','2026-06-21','Uzbekistan','DR Congo'),
    ('K','2026-06-27','Portugal','DR Congo'),
    ('K','2026-06-27','Colombia','Uzbekistan'),
    ('L','2026-06-17','England','Panama'),
    ('L','2026-06-18','Croatia','Ghana'),
    ('L','2026-06-21','England','Croatia'),
    ('L','2026-06-22','Panama','Ghana'),
    ('L','2026-06-27','England','Ghana'),
    ('L','2026-06-27','Croatia','Panama'),
]

upcoming_rows = []
for i, (grp, date_str, home, away) in enumerate(SCHEDULE_2026):
    if i % 12 == 0: print(f"  Match {i+1}/{len(SCHEDULE_2026)}...")
    date = pd.Timestamp(date_str)
    r    = build_features(home, away, date)
    prob = float(np.clip(final_model.predict(pd.DataFrame([{c: r[c] for c in FEATURE_COLS}]))[0], 0, 1))
    upcoming_rows.append({
        'group':        f'Group {grp}',
        'date':         date_str,
        'home_team':    home,
        'away_team':    away,
        'home_win_pct': round(prob * 100, 1),
        'away_win_pct': round((1 - prob) * 100, 1),
        'prediction':   'Home Win' if prob >= 0.5 else 'Away Win',
        'home_rank':    get_rank(home, date),
        'away_rank':    get_rank(away, date),
    })

upcoming_df = pd.DataFrame(upcoming_rows)
upcoming_df.to_csv('data/upcoming_2026.csv', index=False)

print(f"\nAll done!")
print(f"Brier score: {overall_brier:.4f}")
print(f"Accuracy (decisive): {overall_acc:.1%}")
for f in ['data/model.pkl','data/upcoming_2026.csv',
          'data/predictions.csv','data/fifa_ranking-2023-07-20.csv']:
    if os.path.exists(f):
        print(f"  {f}  ({os.path.getsize(f)//1024} KB)")

print("\nSouth Korea vs Czech Republic preview:")
sk_row = upcoming_df[(upcoming_df['home_team']=='South Korea') &
                     (upcoming_df['away_team']=='Czech Republic')]
if len(sk_row):
    r = sk_row.iloc[0]
    print(f"  South Korea: {r['home_win_pct']}%  |  Czech Republic: {r['away_win_pct']}%")

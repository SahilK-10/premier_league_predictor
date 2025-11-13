import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

SEASONS_CSV = [
    ('2223.csv', 2023),
    ('2324.csv', 2024),
    ('2425.csv', 2025),
    ('2526.csv', 2026),
]
GW_TO_PREDICT = 12


GW12_FIXTURES = [
    ('Burnley', 'Chelsea'), ('Bournemouth', 'West Ham'),
    ('Brighton', 'Brentford'), ('Fulham', 'Sunderland'),
    ('Liverpool', 'Nottingham Forest'), ('Wolves', 'Crystal Palace'),
    ('Newcastle', 'Man City'), ('Leeds', 'Aston Villa'),
    ('Arsenal', 'Tottenham'), ('Man Utd', 'Everton')
]


FEATURES_OF_INTEREST = [
    'gameweek', 'HomeTeam', 'AwayTeam',
    'FTHG', 'FTAG', 'HTHG', 'HTAG',
    'HS', 'AS', 'HST', 'AST',
    'HF', 'AF', 'HC', 'AC',
    'HY', 'AY', 'HR', 'AR'
]


def load_data():
    frames = []
    for filename, season in SEASONS_CSV:
        df = pd.read_csv(filename)
        cols = [c for c in FEATURES_OF_INTEREST if c in df.columns]
        df = df[cols]
        df['season'] = season
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df['gameweek'] = pd.to_numeric(
        df['gameweek'], errors='coerce').dropna().astype(int)
    stat_cols = ['HS', 'AS', 'HST', 'AST', 'HF',
                 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']
    for c in stat_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    return df


def prepare(df):
    historical = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique().tolist()
    future = [t for fixture in GW12_FIXTURES for t in fixture]
    all_teams = list(set(historical + future))
    le = LabelEncoder()
    le.fit(all_teams)
    df['Home_enc'] = le.transform(df['HomeTeam'])
    df['Away_enc'] = le.transform(df['AwayTeam'])
    feature_cols = [c for c in df.columns if c not in [
        'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'season']]
    df[feature_cols] = df[feature_cols].fillna(0)
    return df, feature_cols, le


def train(df, features):
    latest = df['season'].max()
    train = df[~((df.season == latest) & (df.gameweek >= GW_TO_PREDICT))]
    test = df[(df.season == latest) & (df.gameweek >= GW_TO_PREDICT)]
    Xtr = train[features]
    yh = train['FTHG']
    ya = train['FTAG']
    models = {}
    xh = xgb.XGBRegressor(n_estimators=200, max_depth=6,
                          random_state=42, verbosity=0)
    xa = xgb.XGBRegressor(n_estimators=200, max_depth=6,
                          random_state=42, verbosity=0)
    rf_h = RandomForestRegressor(
        n_estimators=200, max_depth=10, random_state=42)
    rf_a = RandomForestRegressor(
        n_estimators=200, max_depth=10, random_state=42)
    xh.fit(Xtr, yh)
    xa.fit(Xtr, ya)
    rf_h.fit(Xtr, yh)
    rf_a.fit(Xtr, ya)
    models['xh'] = xh
    models['xa'] = xa
    models['rfh'] = rf_h
    models['rfa'] = rf_a
    if not test.empty:
        Xte = test[features]
        yh_t = test['FTHG']
        ya_t = test['FTAG']
        ph = (xh.predict(Xte)+rf_h.predict(Xte))/2
        pa = (xa.predict(Xte)+rf_a.predict(Xte))/2
        print('MAE home:', mean_absolute_error(yh_t, ph))
        print('MAE away:', mean_absolute_error(ya_t, pa))
    return models


def predict(models, df, features, le):
    fixtures = GW12_FIXTURES
    rows = []
    latest = df['season'].max()
    recent = df[df['season'] == latest]
    for home, away in fixtures:
        hr = recent[recent['HomeTeam'] == home].tail(3)
        ar = recent[recent['AwayTeam'] == away].tail(3)
        row = {'gameweek': GW_TO_PREDICT, 'Home_enc': le.transform(
            [home])[0], 'Away_enc': le.transform([away])[0]}
        for c in FEATURES_OF_INTEREST:
            if c in ['HomeTeam', 'AwayTeam', 'gameweek']:
                continue
            if c in hr.columns:
                row[c] = hr[c].mean()
            elif c in ar.columns:
                row[c] = ar[c].mean()
        rows.append(row)
    pf = pd.DataFrame(rows).fillna(0)
    Xp = pf[features]
    ph = (models['xh'].predict(Xp)+models['rfh'].predict(Xp))/2
    pa = (models['xa'].predict(Xp)+models['rfa'].predict(Xp))/2
    ph = np.clip(ph, 0, 5)
    pa = np.clip(pa, 0, 5)
    for (h, a), hpr, apr in zip(fixtures, ph, pa):
        print(f"{h} {int(round(hpr))}-{int(round(apr))} {a}")


def main():
    df = load_data()
    df, feature_cols, le = prepare(df)
    models = train(df, feature_cols)
    predict(models, df, feature_cols, le)


if __name__ == '__main__':
    main()

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Config
WINDOW = 5
LONG_WINDOW = 10
SEASONS_CSV = [
    ('2223.csv', 2023),
    ('2324.csv', 2024),
    ('2425.csv', 2025),
    ('2526.csv', 2026),
]
GW_TO_PREDICT = 4

def load_and_format_csv(filename, season_end_year):
    df = pd.read_csv(filename)
    df = df.rename(columns={
        'Date': 'date',
        'HomeTeam': 'homeTeam',
        'AwayTeam': 'awayTeam',
        'FTHG': 'homeGoals',
        'FTAG': 'awayGoals',
        'Gameweek': 'gameweek'
    })
    df = df[['date', 'homeTeam', 'awayTeam', 'homeGoals', 'awayGoals', 'gameweek']].copy()
    df['season'] = season_end_year
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    df['gameweek'] = pd.to_numeric(df['gameweek'], errors='coerce')
    return df

def prepare_data():
    dfs = [load_and_format_csv(f, y) for f, y in SEASONS_CSV]
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=['gameweek'])
    df['gameweek'] = df['gameweek'].astype(int)
    return df

def calculate_team_strength(df):
    team_stats = {}
    teams = pd.concat([df['homeTeam'], df['awayTeam']]).unique()
    for team in teams:
        home = df[df['homeTeam']==team]
        away = df[df['awayTeam']==team]
        team_stats[team] = {
            'home_attack': home['homeGoals'].mean(),
            'home_defense': home['awayGoals'].mean(),
            'away_attack': away['awayGoals'].mean(),
            'away_defense': away['homeGoals'].mean()
        }
    return team_stats

def engineer_enhanced_features(df):
    df['home_pts'] = (df.homeGoals>df.awayGoals).map({True:3,False:0}) + (df.homeGoals==df.awayGoals).astype(int)
    df['away_pts'] = (df.awayGoals>df.homeGoals).map({True:3,False:0}) + (df.homeGoals==df.awayGoals).astype(int)
    df = df.sort_values(['season','date']).reset_index(drop=True)

    for w,label in [(WINDOW,'short'),(LONG_WINDOW,'long')]:
        df[f'home_form_{label}'] = df.groupby('homeTeam')['home_pts'].rolling(w,min_periods=1).mean().reset_index(level=0,drop=True)
        df[f'away_form_{label}'] = df.groupby('awayTeam')['away_pts'].rolling(w,min_periods=1).mean().reset_index(level=0,drop=True)

    df['home_gf'] = df.groupby('homeTeam')['homeGoals'].rolling(WINDOW,min_periods=1).mean().reset_index(level=0,drop=True)
    df['home_ga'] = df.groupby('homeTeam')['awayGoals'].rolling(WINDOW,min_periods=1).mean().reset_index(level=0,drop=True)
    df['away_gf'] = df.groupby('awayTeam')['awayGoals'].rolling(WINDOW,min_periods=1).mean().reset_index(level=0,drop=True)
    df['away_ga'] = df.groupby('awayTeam')['homeGoals'].rolling(WINDOW,min_periods=1).mean().reset_index(level=0,drop=True)

    df['home_gd'] = df['homeGoals'] - df['awayGoals']
    df['away_gd'] = df['awayGoals'] - df['homeGoals']
    df['home_gd_form'] = df.groupby('homeTeam')['home_gd'].rolling(WINDOW,min_periods=1).mean().reset_index(level=0,drop=True)
    df['away_gd_form'] = df.groupby('awayTeam')['away_gd'].rolling(WINDOW,min_periods=1).mean().reset_index(level=0,drop=True)

    team_stats = calculate_team_strength(df)
    df['ha_strength'] = df['homeTeam'].map(lambda t: team_stats[t]['home_attack'])
    df['hd_strength'] = df['homeTeam'].map(lambda t: team_stats[t]['home_defense'])
    df['aa_strength'] = df['awayTeam'].map(lambda t: team_stats[t]['away_attack'])
    df['ad_strength'] = df['awayTeam'].map(lambda t: team_stats[t]['away_defense'])

    df = pd.get_dummies(df, columns=['homeTeam','awayTeam'], drop_first=False)
    drop_cols = ['home_pts','away_pts','date','home_gd','away_gd']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return df, team_stats

def split_train_test(df, gw_cutoff=GW_TO_PREDICT):
    latest = df['season'].max()
    train = df[~((df.season==latest)&(df.gameweek>=gw_cutoff))].copy()
    test  = df[(df.season==latest)&(df.gameweek>=gw_cutoff)].copy()
    if test.empty:
        gw_cutoff=3
        train = df[~((df.season==latest)&(df.gameweek>=gw_cutoff))].copy()
        test  = df[(df.season==latest)&(df.gameweek>=gw_cutoff)].copy()
    feat = [c for c in df.columns if c not in ['season','homeGoals','awayGoals','gameweek']]
    Xtr, Yhtr, Yatr = train[feat], train['homeGoals'], train['awayGoals']
    Xte = test[feat] if not test.empty else pd.DataFrame(columns=feat)
    Yhte = test['homeGoals'] if not test.empty else pd.Series(dtype=int)
    Yate = test['awayGoals'] if not test.empty else pd.Series(dtype=int)
    return Xtr, Xte, Yhtr, Yatr, Yhte, Yate, feat

def train_enhanced_model(Xtr, Ytr, Xte, Yte, label):
    Xtr = Xtr.fillna(0)
    Xte = Xte.fillna(0)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    models = [
        GradientBoostingRegressor(n_estimators=150,max_depth=6,learning_rate=0.1,random_state=42),
        RandomForestRegressor(n_estimators=150,max_depth=10,random_state=42)
    ]
    preds = []
    for m in models:
        m.fit(Xtr_s, Ytr)
        if not Xte_s.size: continue
        preds.append(m.predict(Xte_s))
    if preds:
        ens = np.mean(preds,axis=0)
        print(f"{label} MAE:", mean_absolute_error(Yte,ens))
    return models[0], scaler

def create_input(fixtures, df, df_features, team_stats, feat_cols):
    rows=[]
    latest = df['season'].max()
    latest_df = df_features[df_features['season']==latest]
    for home, away in fixtures:
        row = dict.fromkeys(feat_cols,0)
        hdf = latest_df[latest_df[f'homeTeam_{home}']==1] if f'homeTeam_{home}' in latest_df else pd.DataFrame()
        adf = latest_df[latest_df[f'awayTeam_{away}']==1] if f'awayTeam_{away}' in latest_df else pd.DataFrame()
        for f in ['home_form_short','home_form_long','home_gf','home_ga','home_gd_form']:
            row[f] = hdf[f].iloc[-1] if f in hdf and not hdf.empty else 0
        for f in ['away_form_short','away_form_long','away_gf','away_ga','away_gd_form']:
            row[f] = adf[f].iloc[-1] if f in adf and not adf.empty else 0
        row['ha_strength']=team_stats[home]['home_attack'] if home in team_stats else 0
        row['hd_strength']=team_stats[home]['home_defense'] if home in team_stats else 0
        row['aa_strength']=team_stats[away]['away_attack'] if away in team_stats else 0
        row['ad_strength']=team_stats[away]['away_defense'] if away in team_stats else 0
        if f'homeTeam_{home}' in row: row[f'homeTeam_{home}']=1
        if f'awayTeam_{away}' in row: row[f'awayTeam_{away}']=1
        rows.append(row)
    return pd.DataFrame(rows)[feat_cols]

def main():
    fixtures =[
    ('Arsenal', 'Nottingham Forest'),
    ('Bournemouth', 'Brighton'),
    ('Crystal Palace', 'Sunderland'),
    ('Everton', 'Aston Villa'),
    ('Fulham', 'Leeds'),
    ('Newcastle', 'Wolves'),
    ('West Ham', 'Spurs'),
    ('Brentford', 'Chelsea'),
    ('Burnley', 'Liverpool'),
    ('Man City', 'Man Utd'),
]

    df = prepare_data()
    df_feat, team_stats = engineer_enhanced_features(df)
    df_feat['gameweek'], df_feat['season'] = df['gameweek'], df['season']
    df_feat['homeGoals'], df_feat['awayGoals'] = df['homeGoals'], df['awayGoals']

    Xtr,Xte,Yhtr,Yatr,Yhte,Yate,feat_cols = split_train_test(df_feat)
    home_model,homescaler = train_enhanced_model(Xtr,Yhtr,Xte,Yhte,"Home Goals")
    away_model,awayscaler = train_enhanced_model(Xtr,Yatr,Xte,Yate,"Away Goals")
    Xpred = create_input(fixtures,df,df_feat,team_stats,feat_cols).fillna(0)

    if not Xpred.empty:
        hps = home_model.predict(homescaler.transform(Xpred))
        aps = away_model.predict(homescaler.transform(Xpred))
        hps,aps = np.clip(hps,0,5),np.clip(aps,0,5)
        print("=== PREDICTIONS FOR GW4===")
        for (h, a), hp, ap in zip(fixtures, hps, aps):
            print(f"{h} {int(round(hp))} - {int(round(ap))} {a}")



if __name__=="__main__":
    main()


import pandas as pd
import numpy as np
import requests
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from pathlib import Path
from typing import Optional

# ---------- config ----------
BASE_URL = "https://fantasy.premierleague.com/api"
CACHE_DIR = Path("player_cache")
CACHE_DIR.mkdir(exist_ok=True)
DEFAULT_FDR = 5
RANDOM_STATE = 42

session = requests.Session()

# ---------- data helpers ----------
def get_json(endpoint: str):
    r = session.get(f"{BASE_URL}/{endpoint}")
    r.raise_for_status()
    return r.json()

def get_bootstrap():
    return get_json("bootstrap-static/")

def get_fixtures():
    return pd.DataFrame(get_json("fixtures/"))

def get_player_history(player_id: int, cache_dir: Path = CACHE_DIR) -> pd.DataFrame:

    cache_file = cache_dir / f"{player_id}.pkl"

    if cache_file.exists():
        with cache_file.open("rb") as f:
            return pickle.load(f)
        
    url = f"element-summary/{player_id}/"
    data = get_json(url)
    df = pd.DataFrame(data.get("history", []))

    if not df.empty:
        df["player_id"] = player_id
    with cache_file.open("wb") as f:
        pickle.dump(df, f)

    return df

# ---------- feature engineering ----------
def rolling_average(df: pd.DataFrame, group_col: str, value_col: str, new_col: str, window: int = 3, shift: int = 1):
    df[new_col] = (
        df.groupby(group_col)[value_col]
        .rolling(window)
        .mean()
        .shift(shift)
        .reset_index(level=0, drop=True)
    )

def featurize(all_histories: pd.DataFrame, players_df: pd.DataFrame, fixtures_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    # rolling features
    rolling_average(all_histories, "player_id", "total_points", "points_last_3")
    rolling_average(all_histories, "player_id", "ict_index", "ict_last_3")
    rolling_average(all_histories, "player_id", "minutes", "minutes_last_3")

    # target = next gw points
    all_histories["target_next_gw"] = all_histories.groupby("player_id")["total_points"].shift(-1)

    # drop rows without target/features
    all_histories = all_histories.dropna(subset=["points_last_3", "ict_last_3", "minutes_last_3", "target_next_gw"]).copy()

    # get next GW id safely
    next_mask = events_df[["is_current", "is_next"]].any(axis=1)
    next_events = events_df.loc[next_mask, "id"]
    next_gw: Optional[int] = int(next_events.iloc[0]) if not next_events.empty else None

    # filter fixtures for next gw
    if next_gw is not None:
        fixtures_next = fixtures_df[fixtures_df["event"] == next_gw]
    else:
        fixtures_next = pd.DataFrame(columns=fixtures_df.columns)

    # build fixture difficulty mapping (home+away)
    if not fixtures_next.empty:
        home = fixtures_next[["team_h", "team_h_difficulty"]].rename(columns={"team_h": "team", "team_h_difficulty": "fixture_difficulty"})
        away = fixtures_next[["team_a", "team_a_difficulty"]].rename(columns={"team_a": "team", "team_a_difficulty": "fixture_difficulty"})
        fixture_diff = pd.concat([home, away], ignore_index=True)
    else:
        fixture_diff = pd.DataFrame(columns=["team", "fixture_difficulty"])

    # merge player team and fixture difficulty
    all_histories = all_histories.merge(players_df[["id", "team"]], left_on="player_id", right_on="id", how="left")
    all_histories = all_histories.merge(fixture_diff, on="team", how="left")
    all_histories["fixture_difficulty"] = all_histories["fixture_difficulty"].fillna(DEFAULT_FDR)

    return all_histories

# ---------- modeling ----------
def train_model(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, y_pred))
    return model

# ---------- prediction/reporting ----------
def predict_and_report(model, all_histories, players_df, features, my_team_ids):
    latest = all_histories.groupby("player_id").last().reset_index()
    latest["predicted_points"] = model.predict(latest[features])
    latest["pick_score"] = latest["predicted_points"] / latest["predicted_points"].max() * 100
    latest = latest.merge(players_df[["id", "first_name", "second_name"]], left_on="player_id", right_on="id", how="left")

    print("\nTop 15 Recommended Players for Next Gameweek:")
    print(latest.sort_values("pick_score", ascending=False)[["first_name", "second_name", "predicted_points", "pick_score"]].head(15))

    my_team = latest[latest["player_id"].isin(my_team_ids)].sort_values("predicted_points")
    print("\nBench Candidates (lowest predicted points):")
    print(my_team[["first_name", "second_name", "predicted_points", "pick_score"]])


# ---------- get my team ----------

my_team_id = 6167828

def get_latest_team(team_id: int):
    # 1. Get all events
    bs = session.get(f"{BASE_URL}/bootstrap-static/").json()
    events = bs["events"]
    
    # 2. Find the last finished or current GW
    last_gw = max(e["id"] for e in events if e["finished"] or e["is_current"])
    
    # 3. Fetch your picks for that GW
    r = session.get(f"{BASE_URL}/entry/{team_id}/event/{last_gw}/picks/")
    r.raise_for_status()
    data = r.json()
    
    picks = pd.DataFrame(data["picks"])
    return picks[["element", "position", "multiplier"]]

my_team_df = get_latest_team(my_team_id)

# ---------- main ----------
def main():
    bs = get_bootstrap()
    players_df = pd.DataFrame(bs["elements"])
    events_df = pd.DataFrame(bs["events"])
    fixtures_df = get_fixtures()

    histories = pd.concat([get_player_history(pid) for pid in players_df["id"]], ignore_index=True)
    all_hist = featurize(histories, players_df, fixtures_df, events_df)

    features = ["points_last_3", "ict_last_3", "minutes_last_3", "fixture_difficulty"]
    X = all_hist[features]
    y = all_hist["target_next_gw"]

    model = train_model(X, y)

    my_team_ids = my_team_df["element"].tolist()
    predict_and_report(model, all_hist, players_df, features, my_team_ids)

main()


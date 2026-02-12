"""
ESPN Dynasty League Projections App
Integrates ESPN Fantasy Baseball API for roster data and FanGraphs for projections
"""
import io
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

import numpy as np
import pandas as pd
import requests
import streamlit as st
from rapidfuzz import process, fuzz
from scipy.optimize import linear_sum_assignment


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="ESPN Dynasty Projections",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Projection System Configuration
# -----------------------------
HITTING_PROJECTIONS = {
    "Steamer": "steamer",
    "THE BAT X": "thebatx",
    "THE BAT": "thebat",
    "ATC": "atc",
    "OOPSY": "oopsy",
}

PITCHING_PROJECTIONS = {
    "Steamer": "steamer",
    "THE BAT": "thebat",
    "ATC": "atc",
    "OOPSY": "oopsy",
}

# ESPN Eligible-Slot IDs for baseball (from espn-api package)
ESPN_POSITION_MAP = {
    0: "C",
    1: "1B",
    2: "2B",
    3: "3B",
    4: "SS",
    5: "OF",
    6: "MI",    # 2B/SS
    7: "CI",    # 1B/3B
    8: "LF",
    9: "CF",
    10: "RF",
    11: "DH",
    12: "UTIL",
    13: "P",
    14: "SP",
    15: "RP",
    16: "BE",
    17: "IL",
    18: "IL+",
    19: "IF",   # 1B/2B/SS/3B
}

# ESPN defaultPositionId → primary position
ESPN_DEFAULT_POSITION_MAP = {
    1: "P",
    2: "C",
    3: "1B",
    4: "2B",
    5: "3B",
    6: "SS",
    7: "LF",
    8: "CF",
    9: "RF",
    10: "DH",
}

# Slot IDs that represent actual field/playing positions (not roster management slots)
ESPN_REAL_POSITION_SLOTS = {0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 13, 14, 15}


def get_projection_url(proj_type: str, is_pitching: bool = False) -> str:
    """Generate FanGraphs projection URL for the given projection system."""
    stats = "pit" if is_pitching else "bat"
    return f"https://www.fangraphs.com/projections?type={proj_type}&stats={stats}&statgroup=fantasy&fantasypreset=dashboard"


# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #c41e3a 0%, #002d72 100%);
        padding: 1.5rem 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1.5rem;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1a1f2e;
    }
    
    [data-testid="stSidebar"] label {
        color: #e2e8f0 !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown span,
    [data-testid="stSidebar"] .stMarkdown li {
        color: #cbd5e1 !important;
    }
    
    [data-testid="stSidebar"] .stExpander summary span {
        color: #f1f5f9 !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stSidebar"] .stExpander {
        background-color: #242b3d;
        border-radius: 8px;
        border: 1px solid #374151;
    }
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stTextInput label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stCheckbox label {
        color: #e2e8f0 !important;
    }
    
    [data-testid="stSidebar"] .stCaption {
        color: #94a3b8 !important;
    }
    
    [data-testid="stSidebar"] hr {
        border-color: #374151;
    }
    
    /* Sidebar section headers */
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #f8fafc !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 1rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 8px 16px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------
# ESPN API Functions
# -----------------------------
@st.cache_data(ttl=1800)  # Cache for 30 minutes
def fetch_espn_league_info(league_id: str, year: int, espn_s2: str = "", swid: str = "") -> Tuple[Dict, Optional[str]]:
    """Fetch league info from ESPN Fantasy API."""
    base_url = f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/flb/seasons/{year}/segments/0/leagues/{league_id}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
    }
    
    cookies = {}
    if espn_s2 and swid:
        cookies = {
            "espn_s2": espn_s2,
            "SWID": swid if swid.startswith("{") else f"{{{swid}}}"
        }
    
    try:
        params = {"view": "mSettings"}
        response = requests.get(base_url, headers=headers, cookies=cookies, params=params, timeout=15)
        
        if response.status_code == 401:
            return {}, "Authentication required. Please provide ESPN cookies (espn_s2 and SWID)."
        elif response.status_code == 404:
            return {}, "League not found. Check the League ID."
        elif response.status_code != 200:
            return {}, f"ESPN API error: {response.status_code}"
        
        data = response.json()
        
        league_info = {
            "leagueName": data.get("settings", {}).get("name", "ESPN League"),
            "seasonId": data.get("seasonId", year),
            "scoringPeriodId": data.get("scoringPeriodId", 0),
        }
        
        return league_info, None
        
    except requests.exceptions.Timeout:
        return {}, "ESPN API request timed out"
    except requests.exceptions.RequestException as e:
        return {}, f"ESPN API request failed: {str(e)}"
    except json.JSONDecodeError:
        return {}, "Invalid response from ESPN API"


@st.cache_data(ttl=900)  # Cache for 15 minutes
def fetch_espn_rosters(league_id: str, year: int, espn_s2: str = "", swid: str = "") -> Tuple[pd.DataFrame, Optional[str], Optional[Dict]]:
    """Fetch all team rosters from ESPN Fantasy API."""
    base_url = f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/flb/seasons/{year}/segments/0/leagues/{league_id}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
    }
    
    cookies = {}
    if espn_s2 and swid:
        cookies = {
            "espn_s2": espn_s2,
            "SWID": swid if swid.startswith("{") else f"{{{swid}}}"
        }
    
    try:
        # Get teams and rosters
        response = requests.get(
            base_url, 
            headers=headers, 
            cookies=cookies, 
            params={"view": ["mTeam", "mRoster"]}, 
            timeout=15
        )
        
        if response.status_code == 401:
            return pd.DataFrame(), "Authentication required. Please provide ESPN cookies.", None
        elif response.status_code != 200:
            return pd.DataFrame(), f"ESPN API error: {response.status_code}", None
        
        data = response.json()
        
        teams = data.get("teams", [])
        
        # Debug: collect info about all teams to understand the structure
        raw_debug = {
            "keys": list(data.keys()),
            "num_teams_in_response": len(teams),
            "teams_summary": []
        }
        
        roster_rows = []
        
        for team in teams:
            team_id = team.get("id")
            team_name = team.get("name", f"Team {team_id}")
            team_abbrev = team.get("abbrev", "")
            owners = team.get("owners", [])
            roster = team.get("roster", {})
            entries = roster.get("entries", [])
            
            # Add to debug summary
            raw_debug["teams_summary"].append({
                "id": team_id,
                "name": team_name,
                "abbrev": team_abbrev,
                "has_owners": len(owners) > 0,
                "num_roster_entries": len(entries)
            })
            
            # FILTER: Only include teams that have owners (real fantasy teams)
            # Teams without owners are placeholder/template teams
            if not owners:
                continue
            
            for entry in entries:
                player_pool = entry.get("playerPoolEntry", {})
                player = player_pool.get("player", {})
                
                if not player:
                    continue
                
                player_id = player.get("id")
                full_name = player.get("fullName", "Unknown")
                
                # Get position eligibility from eligible roster slots
                eligible_slots = player.get("eligibleSlots", [])
                positions = []
                for slot_id in eligible_slots:
                    if slot_id not in ESPN_REAL_POSITION_SLOTS:
                        continue
                    pos = ESPN_POSITION_MAP.get(slot_id)
                    if pos and pos not in positions:
                        positions.append(pos)
                
                # Use defaultPositionId as fallback if no real positions found
                default_position = player.get("defaultPositionId", None)
                if not positions and default_position:
                    fallback = ESPN_DEFAULT_POSITION_MAP.get(default_position)
                    if fallback:
                        positions = [fallback]
                
                # Current lineup slot
                lineup_slot_id = entry.get("lineupSlotId", 16)
                lineup_slot = ESPN_POSITION_MAP.get(lineup_slot_id, "BE")
                
                # Determine if pitcher (defaultPositionId == 1)
                # is_pitcher = (default_position == 1)

                # # Fallback: if defaultPositionId is missing, infer from eligible slot IDs
                # if default_position is None:
                is_pitcher = any(sid in (13, 14, 15) for sid in eligible_slots)

                roster_rows.append({
                    "Team": team_name,
                    "TeamID": team_id,
                    "Player": full_name,
                    "ESPN_ID": player_id,
                    "Position": ",".join(positions) if positions else (ESPN_DEFAULT_POSITION_MAP.get(default_position, "UTIL") if default_position else "UTIL"),
                    "LineupSlot": lineup_slot,
                    "IsPitcher": is_pitcher,
                })
        
        if not roster_rows:
            return pd.DataFrame(), "No roster data found", raw_debug
        
        df = pd.DataFrame(roster_rows)
        
        # Add final counts to debug
        raw_debug["fantasy_teams_with_owners"] = df["Team"].nunique()
        raw_debug["total_players"] = len(df)
        
        return df, None, raw_debug
        
    except requests.exceptions.Timeout:
        return pd.DataFrame(), "ESPN API request timed out", None
    except requests.exceptions.RequestException as e:
        return pd.DataFrame(), f"ESPN API request failed: {str(e)}", None
    except (json.JSONDecodeError, KeyError) as e:
        return pd.DataFrame(), f"Error parsing ESPN data: {str(e)}", None


# -----------------------------
# FanGraphs API Functions
# -----------------------------
def _convert_fangraphs_api_url(page_url: str) -> str:
    """Convert FanGraphs page URL to API URL."""
    parsed = urlparse(page_url)
    query = parse_qs(parsed.query)
    
    api_params = {
        "pos": "all",
        "lg": "all",
        "stats": query.get("stats", ["bat"])[0],
        "type": query.get("type", ["steamer"])[0],
        "statgroup": "fantasy",
        "fantasypreset": "dashboard",
    }
    
    return "https://www.fangraphs.com/api/projections?" + urlencode(api_params)


@st.cache_data(ttl=21600)  # Cache for 6 hours
def fetch_fangraphs_projections(proj_type: str, is_pitching: bool = False) -> Tuple[pd.DataFrame, Optional[str]]:
    """Fetch projections from FanGraphs API."""
    stats = "pit" if is_pitching else "bat"
    api_url = f"https://www.fangraphs.com/api/projections?pos=all&lg=all&stats={stats}&type={proj_type}&statgroup=fantasy&fantasypreset=dashboard"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; dynasty-projections-app/1.0)",
        "Accept": "application/json",
    }
    
    try:
        response = requests.get(api_url, headers=headers, timeout=30)
        if response.status_code != 200:
            return pd.DataFrame(), f"FanGraphs API error: {response.status_code}"
        
        data = response.json()
        if not data:
            return pd.DataFrame(), "Empty response from FanGraphs"
        
        df = pd.DataFrame(data)
        return df, None
        
    except requests.exceptions.Timeout:
        return pd.DataFrame(), "FanGraphs request timed out"
    except requests.exceptions.RequestException as e:
        return pd.DataFrame(), f"FanGraphs request failed: {str(e)}"
    except (json.JSONDecodeError, ValueError) as e:
        return pd.DataFrame(), f"Error parsing FanGraphs data: {str(e)}"


# -----------------------------
# Name Matching & Processing
# -----------------------------
SUFFIXES = [r"\bJr\.?\b", r"\bSr\.?\b", r"\bII\b", r"\bIII\b", r"\bIV\b", r"\bV\b"]


def _clean_name(name: str) -> str:
    """Normalize a player name for matching."""
    if not isinstance(name, str):
        return ""
    n = name.strip().lower()
    for suf in SUFFIXES:
        n = re.sub(suf, "", n, flags=re.IGNORECASE)
    n = re.sub(r"['\.\-]", "", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def _name_col(df: pd.DataFrame) -> str:
    """Return the best name column for a projection DataFrame, safely."""
    if "Name" in df.columns:
        return "Name"
    if "PlayerName" in df.columns:
        return "PlayerName"
    if len(df.columns) > 0:
        return df.columns[0]
    return "__missing__"


def build_projection_index(bat_df: pd.DataFrame, pit_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build projection DataFrames with clean name index."""
    bat = bat_df.copy()
    pit = pit_df.copy()
    
    name_col_bat = _name_col(bat)
    name_col_pit = _name_col(pit)
    
    if name_col_bat != "__missing__":
        bat["__CleanName"] = bat[name_col_bat].astype(str).apply(_clean_name)
    else:
        bat["__CleanName"] = pd.Series(dtype=str)

    if name_col_pit != "__missing__":
        pit["__CleanName"] = pit[name_col_pit].astype(str).apply(_clean_name)
    else:
        pit["__CleanName"] = pd.Series(dtype=str)
    
    return bat, pit


def match_roster_to_projections(
    roster: pd.DataFrame,
    bat: pd.DataFrame,
    pit: pd.DataFrame,
    threshold: int = 80
) -> pd.DataFrame:
    """Match roster players to FanGraphs projections."""
    roster = roster.copy()
    roster["__CleanName"] = roster["Player"].astype(str).apply(_clean_name)
    
    bat_names = list(bat["__CleanName"].unique())
    pit_names = list(pit["__CleanName"].unique())
    
    matched_names = []
    matched_types = []
    match_scores = []
    
    for _, row in roster.iterrows():
        clean = row["__CleanName"]
        is_pitcher = row.get("IsPitcher", False)
        
        # Check for exact match first
        if is_pitcher and clean in pit_names:
            matched_names.append(clean)
            matched_types.append("Pitcher")
            match_scores.append(100)
            continue
        elif not is_pitcher and clean in bat_names:
            matched_names.append(clean)
            matched_types.append("Hitter")
            match_scores.append(100)
            continue
        
        # Try fuzzy matching
        pool = pit_names if is_pitcher else bat_names
        player_type = "Pitcher" if is_pitcher else "Hitter"
        
        if pool:
            match = process.extractOne(clean, pool, scorer=fuzz.ratio)
            if match and match[1] >= threshold:
                matched_names.append(match[0])
                matched_types.append(player_type)
                match_scores.append(match[1])
                continue
        
        # Try opposite pool as fallback
        alt_pool = bat_names if is_pitcher else pit_names
        alt_type = "Hitter" if is_pitcher else "Pitcher"
        
        if alt_pool:
            match = process.extractOne(clean, alt_pool, scorer=fuzz.ratio)
            if match and match[1] >= threshold:
                matched_names.append(match[0])
                matched_types.append(alt_type)
                match_scores.append(match[1])
                continue
        
        # No match found
        matched_names.append("")
        matched_types.append("")
        match_scores.append(0)
    
    roster["Matched_Name"] = matched_names
    roster["Matched_Type"] = matched_types
    roster["Match_Score"] = match_scores
    
    return roster


def get_near_matches(
    player_name: str,
    is_pitcher: bool,
    bat: pd.DataFrame,
    pit: pd.DataFrame,
    n: int = 8,
) -> List[Tuple[str, str, int]]:
    """Return top-N fuzzy matches for a player across both pools.
    
    Returns list of (clean_name, display_name, score) tuples.
    """
    clean = _clean_name(player_name)
    results = []

    # Determine name columns
    bat_name_col = _name_col(bat)
    pit_name_col = _name_col(pit)

    # Build lookup: clean_name -> display_name
    bat_display = {}
    for _, r in bat.iterrows():
        cn = r.get("__CleanName", _clean_name(str(r.get(bat_name_col, ""))))
        bat_display[cn] = str(r.get(bat_name_col, cn))
    pit_display = {}
    for _, r in pit.iterrows():
        cn = r.get("__CleanName", _clean_name(str(r.get(pit_name_col, ""))))
        pit_display[cn] = str(r.get(pit_name_col, cn))

    # Search primary pool first, then secondary
    primary_pool = list(pit_display.keys()) if is_pitcher else list(bat_display.keys())
    primary_disp = pit_display if is_pitcher else bat_display
    primary_type = "Pitcher" if is_pitcher else "Hitter"

    secondary_pool = list(bat_display.keys()) if is_pitcher else list(pit_display.keys())
    secondary_disp = bat_display if is_pitcher else pit_display
    secondary_type = "Hitter" if is_pitcher else "Pitcher"

    if primary_pool:
        matches = process.extract(clean, primary_pool, scorer=fuzz.ratio, limit=n)
        for m_name, m_score, _ in matches:
            disp = primary_disp.get(m_name, m_name)
            results.append((m_name, f"{disp} ({primary_type})", m_score, primary_type))

    if secondary_pool:
        matches = process.extract(clean, secondary_pool, scorer=fuzz.ratio, limit=max(2, n // 2))
        for m_name, m_score, _ in matches:
            disp = secondary_disp.get(m_name, m_name)
            results.append((m_name, f"{disp} ({secondary_type})", m_score, secondary_type))

    # Sort by score desc and deduplicate
    results.sort(key=lambda x: -x[2])
    seen = set()
    deduped = []
    for r in results:
        if r[0] not in seen:
            seen.add(r[0])
            deduped.append(r)
    return deduped[:n]


def apply_match_overrides(
    roster_matched: pd.DataFrame,
    overrides: Dict[str, Tuple[str, str]],
) -> pd.DataFrame:
    """Apply manual match overrides to roster_matched DataFrame.
    
    overrides: {player_name: (matched_clean_name, matched_type)}
    """
    r = roster_matched.copy()
    for i, row in r.iterrows():
        pname = str(row.get("Player", ""))
        if pname in overrides:
            clean_match, match_type = overrides[pname]
            if clean_match:  # non-empty means override; empty means "unmatch"
                r.loc[i, "Matched_Name"] = clean_match
                r.loc[i, "Matched_Type"] = match_type
                r.loc[i, "Match_Score"] = 100  # manual override = perfect score
            else:
                r.loc[i, "Matched_Name"] = ""
                r.loc[i, "Matched_Type"] = ""
                r.loc[i, "Match_Score"] = 0
    return r


def add_projection_stats(roster_matched: pd.DataFrame, bat: pd.DataFrame, pit: pd.DataFrame) -> pd.DataFrame:
    """Add projection stats to matched roster."""
    r = roster_matched.copy()
    r["__CleanMatched"] = r["Matched_Name"].astype(str).map(_clean_name)
    bat_copy = bat.copy()
    
    # Derive TB for hitters if not already present
    if "TB" not in bat_copy.columns:
        h = pd.to_numeric(bat_copy.get("H", 0), errors="coerce").fillna(0)
        hr = pd.to_numeric(bat_copy.get("HR", 0), errors="coerce").fillna(0)
        doubles = pd.to_numeric(bat_copy.get("2B", bat_copy.get("X2B", 0)), errors="coerce").fillna(0)
        triples = pd.to_numeric(bat_copy.get("3B", bat_copy.get("X3B", 0)), errors="coerce").fillna(0)
        singles = h - doubles - triples - hr
        bat_copy["TB"] = singles + 2 * doubles + 3 * triples + 4 * hr
    
    bat_idx = bat_copy.set_index("__CleanName", drop=False)
    
    # Derive pitching stats if not already present
    pit_copy = pit.copy()
    if "K" not in pit_copy.columns and "SO" in pit_copy.columns:
        pit_copy["K"] = pit_copy["SO"]
    if "SVHD" not in pit_copy.columns:
        sv = pd.to_numeric(pit_copy.get("SV", 0), errors="coerce").fillna(0)
        hld = pd.to_numeric(pit_copy.get("HLD", pit_copy.get("HD", 0)), errors="coerce").fillna(0)
        pit_copy["SVHD"] = 2 * sv + hld
    if "K/BB" not in pit_copy.columns:
        p_k = pd.to_numeric(pit_copy.get("K", pit_copy.get("SO", 0)), errors="coerce").fillna(0)
        p_bb = pd.to_numeric(pit_copy.get("BB", 0), errors="coerce").fillna(0)
        pit_copy["K/BB"] = np.where(p_bb > 0, p_k / p_bb, np.nan)

    pit_idx = pit_copy.set_index("__CleanName", drop=False)
    
    out_rows = []
    for _, row in r.iterrows():
        cm = row["__CleanMatched"]
        if cm and row.get("Matched_Type") == "Hitter" and cm in bat_idx.index:
            fg_row = bat_idx.loc[cm]
            if isinstance(fg_row, pd.DataFrame):
                fg_row = fg_row.iloc[0]
            # Merge projection columns without overwriting roster columns like 'Team' (fantasy team)
            roster_dict = row.to_dict()
            fg_dict = fg_row.to_dict()
            # FanGraphs uses 'Team' for MLB team; keep fantasy team in roster and store MLB team separately
            if "Team" in fg_dict:
                fg_dict["MLB_Team"] = fg_dict.pop("Team")
            # Avoid confusion between roster 'Player' and FanGraphs 'Name'
            if "Name" in fg_dict:
                fg_dict["FG_Name"] = fg_dict.pop("Name")
            # Drop any remaining keys that would overwrite roster fields
            fg_dict = {k: v for k, v in fg_dict.items() if k not in roster_dict}
            merged = {**roster_dict, **fg_dict}
        elif cm and row.get("Matched_Type") == "Pitcher" and cm in pit_idx.index:
            fg_row = pit_idx.loc[cm]
            if isinstance(fg_row, pd.DataFrame):
                fg_row = fg_row.iloc[0]
            # Merge projection columns without overwriting roster columns like 'Team' (fantasy team)
            roster_dict = row.to_dict()
            fg_dict = fg_row.to_dict()
            # FanGraphs uses 'Team' for MLB team; keep fantasy team in roster and store MLB team separately
            if "Team" in fg_dict:
                fg_dict["MLB_Team"] = fg_dict.pop("Team")
            # Avoid confusion between roster 'Player' and FanGraphs 'Name'
            if "Name" in fg_dict:
                fg_dict["FG_Name"] = fg_dict.pop("Name")
            # Drop any remaining keys that would overwrite roster fields
            fg_dict = {k: v for k, v in fg_dict.items() if k not in roster_dict}
            merged = {**roster_dict, **fg_dict}
        else:
            merged = row.to_dict()
        out_rows.append(merged)
    
    return pd.DataFrame(out_rows)


# -----------------------------
# Helper Functions
# -----------------------------
def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _col_series(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series(default, index=df.index)


def _num_col(df: pd.DataFrame, col: str, default=0.0) -> pd.Series:
    return _to_num(_col_series(df, col)).fillna(default)


def _split_positions(pos_raw: str) -> List[str]:
    """Split position string into list of positions."""
    if not isinstance(pos_raw, str):
        return []
    return [p.strip() for p in re.split(r"[,/]", pos_raw) if p.strip()]


# -----------------------------
# Scoring & Aggregation
# -----------------------------
def compute_zscores(df: pd.DataFrame, cols: List[str], invert: Optional[set] = None) -> pd.DataFrame:
    invert = invert or set()
    z = df.copy()
    for c in cols:
        s = _to_num(_col_series(z, c, default=np.nan))
        mu = np.nanmean(s)
        sd = np.nanstd(s)
        if sd == 0 or np.isnan(sd):
            z[c + "_z"] = np.nan
            continue
        zz = (s - mu) / sd
        if c in invert:
            zz = -zz
        z[c + "_z"] = zz
    return z


def team_totals_from_lineup(
    lineup_hit: pd.DataFrame,
    lineup_pit: pd.DataFrame,
    hit_cats: List[str],
    pit_cats: List[str],
) -> Dict[str, float]:
    """Calculate team totals from lineup projections with proper rate stat calculations."""
    out: Dict[str, float] = {}
    
    # === HITTING STATS ===
    h_AB = _num_col(lineup_hit, "AB", default=0.0).sum()
    h_PA = _num_col(lineup_hit, "PA", default=0.0).sum()
    h_H = _num_col(lineup_hit, "H", default=0.0).sum()
    h_BB = _num_col(lineup_hit, "BB", default=0.0).sum()
    h_HBP = _num_col(lineup_hit, "HBP", default=0.0).sum()
    h_SF = _num_col(lineup_hit, "SF", default=0.0).sum()
    
    # Check for doubles, triples
    h_2B = 0.0
    for col in ["2B", "X2B", "Doubles"]:
        if col in lineup_hit.columns:
            h_2B = _num_col(lineup_hit, col, default=0.0).sum()
            break
    
    h_3B = 0.0
    for col in ["3B", "X3B", "Triples"]:
        if col in lineup_hit.columns:
            h_3B = _num_col(lineup_hit, col, default=0.0).sum()
            break
    
    h_HR = _num_col(lineup_hit, "HR", default=0.0).sum()
    h_1B = max(0.0, h_H - h_2B - h_3B - h_HR)
    
    # Calculate TB
    if "TB" in lineup_hit.columns:
        h_TB = _num_col(lineup_hit, "TB", default=0.0).sum()
    else:
        h_TB = h_1B + 2*h_2B + 3*h_3B + 4*h_HR
    
    for c in hit_cats:
        if c == "AVG":
            out[c] = float(h_H / h_AB) if h_AB > 0 else np.nan
        elif c == "OBP":
            numerator = h_H + h_BB + h_HBP
            denominator = h_AB + h_BB + h_HBP + h_SF
            if denominator == 0 and h_PA > 0:
                denominator = h_PA
            out[c] = float(numerator / denominator) if denominator > 0 else np.nan
        elif c == "SLG":
            out[c] = float(h_TB / h_AB) if h_AB > 0 else np.nan
        elif c == "OPS":
            obp_num = h_H + h_BB + h_HBP
            obp_denom = h_AB + h_BB + h_HBP + h_SF
            if obp_denom == 0 and h_PA > 0:
                obp_denom = h_PA
            obp = float(obp_num / obp_denom) if obp_denom > 0 else 0.0
            slg = float(h_TB / h_AB) if h_AB > 0 else 0.0
            out[c] = obp + slg if (obp_denom > 0 or h_AB > 0) else np.nan
        elif c == "TB":
            out[c] = float(h_TB)
        else:
            out[c] = float(_num_col(lineup_hit, c, default=0.0).sum())
    
    # === PITCHING STATS ===
    p_IP = _num_col(lineup_pit, "IP", default=0.0).sum()
    p_ER = _num_col(lineup_pit, "ER", default=0.0).sum()
    p_H = _num_col(lineup_pit, "H", default=0.0).sum()
    p_BB = _num_col(lineup_pit, "BB", default=0.0).sum()
    p_K = _num_col(lineup_pit, "K", default=0.0).sum()
    if p_K == 0.0:
        p_K = _num_col(lineup_pit, "SO", default=0.0).sum()
    
    for c in pit_cats:
        if c == "ERA":
            out[c] = float((p_ER * 9) / p_IP) if p_IP > 0 else np.nan
        elif c == "WHIP":
            out[c] = float((p_H + p_BB) / p_IP) if p_IP > 0 else np.nan
        elif c == "K/BB":
            out[c] = float(p_K / p_BB) if p_BB > 0 else np.nan
        else:
            out[c] = float(_num_col(lineup_pit, c, default=0.0).sum())
    
    return out


def team_scores_from_totals(team_totals: pd.DataFrame, cats: List[str], weights: Dict[str, float]) -> pd.DataFrame:
    invert = {"ERA", "WHIP"}
    z = compute_zscores(team_totals, cats, invert=invert).copy()
    z["Team_Score"] = 0.0
    for c in cats:
        z["Team_Score"] += z.get(c + "_z", 0.0).fillna(0.0) * float(weights.get(c, 1.0))
    return z


def calculate_team_category_strength(team_totals: pd.DataFrame, cats: List[str]) -> pd.DataFrame:
    """Calculate each team's relative strength in each category."""
    strength = pd.DataFrame(index=team_totals.index)
    
    for cat in cats:
        if cat not in team_totals.columns:
            continue
        vals = team_totals[cat]
        mu = vals.mean()
        sd = vals.std()
        if sd > 0:
            z = (vals - mu) / sd
            if cat in {"ERA", "WHIP"}:
                z = -z
            strength[cat] = z
        else:
            strength[cat] = 0.0
    
    return strength


# -----------------------------
# Lineup Optimization
# -----------------------------
def optimize_lineup(players: pd.DataFrame, value_col: str, slots: List[str], bench_slots: int = 10) -> Tuple[pd.DataFrame, float]:
    """Optimize lineup assignment using Hungarian algorithm."""
    players = players.copy().reset_index(drop=True)
    players["Value"] = _num_col(players, value_col, default=0.0)
    players["Assigned_Slot"] = "BENCH"
    
    n_players = len(players)
    if n_players == 0:
        return players, 0.0
    
    real_slots = list(slots)
    if len(real_slots) == 0:
        return players.sort_values("Value", ascending=False), 0.0
    
    overflow_n = max(0, n_players - (len(real_slots) + bench_slots))
    all_slots = real_slots + [f"BENCH_{i+1}" for i in range(bench_slots)] + [f"OVERFLOW_{i+1}" for i in range(overflow_n)]
    
    M = 1e9
    # Prefer filling starter slots over bench even if a player's Value is negative.
    # We do this by giving every real (non-bench) slot a large bonus (lower cost),
    # so bench is only chosen when there are more players than real slots.
    BENCH_COST = 0.0
    STARTER_BONUS = 1000.0
    cost = np.full((n_players, len(all_slots)), M, dtype=float)
    
    for i, row in players.iterrows():
        positions = _split_positions(str(row.get("Position", "")))
        value = float(row["Value"])
        
        for j, slot in enumerate(all_slots):
            if slot.startswith("BENCH_") or slot.startswith("OVERFLOW_"):
                cost[i, j] = BENCH_COST
            elif slot == "UTIL" or slot == "DH":
                cost[i, j] = -value - STARTER_BONUS
            elif slot in positions:
                cost[i, j] = -value - STARTER_BONUS
            # elif slot == "P" and any(p in ["SP", "RP", "P"] for p in positions):
            #     cost[i, j] = -value - STARTER_BONUS
            elif slot == "SP" and ("SP" in positions):
                cost[i, j] = -value - STARTER_BONUS
            elif slot == "RP" and ("RP" in positions):
                cost[i, j] = -value - STARTER_BONUS
            elif slot == "OF" and any(p in ["LF", "CF", "RF", "OF"] for p in positions):
                cost[i, j] = -value - STARTER_BONUS
            # elif slot in ("LF", "CF", "RF") and "OF" in positions:
            #     cost[i, j] = -value - STARTER_BONUS
            elif slot == "MI" and any(p in ["SS", "2B"] for p in positions):
                cost[i, j] = -value - STARTER_BONUS
            elif slot == "CI" and any(p in ["1B", "3B"] for p in positions):
                cost[i, j] = -value - STARTER_BONUS
    
    row_ind, col_ind = linear_sum_assignment(cost)
    
    for r_i, c_i in zip(row_ind, col_ind):
        players.loc[r_i, "Assigned_Slot"] = all_slots[c_i]
    
    in_lineup = players["Assigned_Slot"].isin(real_slots)
    lineup_value = float(players.loc[in_lineup, "Value"].sum())
    
    return players.sort_values(by=["Assigned_Slot", "Value"], ascending=[True, False]), lineup_value


# -----------------------------
# Positional Value & Trade Matching
# -----------------------------
POSITION_GROUPS = {
    "C": ["C"],
    "1B": ["1B"],
    "2B": ["2B"],
    "3B": ["3B"],
    "SS": ["SS"],
    "OF": ["LF", "CF", "RF", "OF"],
    "SP": ["SP"],
    "RP": ["RP"],
}


def compute_positional_value(
    scored: pd.DataFrame,
    teams: List[str],
) -> pd.DataFrame:
    """Compute total player value per position group per team (starters + bench).

    Returns a DataFrame with teams as rows and position groups as columns.
    Each player is assigned to exactly one position group using their primary
    (first listed) position.
    """
    rows = []
    for t in teams:
        t_rows = scored[scored["Team"] == t].copy()
        group_vals: Dict[str, float] = {g: 0.0 for g in POSITION_GROUPS}
        group_counts: Dict[str, int] = {g: 0 for g in POSITION_GROUPS}

        for _, player in t_rows.iterrows():
            positions = _split_positions(str(player.get("Position", "")))
            value = float(player.get("Value", 0.0))
            is_pitcher = bool(player.get("IsPitcher", False))

            assigned = False
            # Try to assign to the most specific matching group
            for pos in positions:
                for grp, grp_pos_list in POSITION_GROUPS.items():
                    if pos in grp_pos_list:
                        group_vals[grp] += value
                        group_counts[grp] += 1
                        assigned = True
                        break
                if assigned:
                    break

            # Fallback: pitchers go to SP, hitters unassigned go to a generic bucket
            if not assigned:
                if is_pitcher:
                    # Check if they have P eligibility → split into SP bucket
                    group_vals["SP"] += value
                    group_counts["SP"] += 1
                # Hitters with no matching group are simply not grouped (e.g. DH/UTIL only)

        row = {"Team": t}
        for g in POSITION_GROUPS:
            row[f"{g}_Value"] = group_vals[g]
            row[f"{g}_Count"] = group_counts[g]
        rows.append(row)

    return pd.DataFrame(rows).set_index("Team")


def find_trade_matches(
    pos_values: pd.DataFrame,
    team_strengths: pd.DataFrame,
    my_team: str,
    cats: List[str],
    top_n: int = 5,
) -> List[Dict]:
    """Find the best trade partners for *my_team*.

    Strategy:
    - Identify my_team's weakest position groups (lowest z-scored positional value).
    - Identify my_team's strongest position groups (surplus).
    - For every other team, score how well they complement:
      a) They are strong where I am weak (they can supply what I need).
      b) They are weak where I am strong (they need what I can offer).
    - Also track category-level complementarity so the UI can always show
      actionable targets even when both teams are positionally balanced.
    - Return ranked list of trade partners with suggested position swaps AND
      category-level details.
    """
    groups = list(POSITION_GROUPS.keys())
    value_cols = [f"{g}_Value" for g in groups]

    # Z-score each position group value across the league
    pos_z = pos_values[value_cols].copy()
    for c in value_cols:
        mu = pos_z[c].mean()
        sd = pos_z[c].std()
        pos_z[c] = (pos_z[c] - mu) / sd if sd > 0 else 0.0

    my_z = pos_z.loc[my_team]

    results = []
    for other_team in pos_z.index:
        if other_team == my_team:
            continue
        other_z = pos_z.loc[other_team]

        # Find positions where I'm weak and they're strong
        i_need = []   # positions I'm weak at
        they_need = []  # positions they're weak at
        i_give = []   # positions I'm strong at
        they_give = []  # positions they're strong at

        for g in groups:
            col = f"{g}_Value"
            my_val = my_z[col]
            their_val = other_z[col]

            if my_val < -0.3:
                i_need.append((g, my_val))
            if my_val > 0.3:
                i_give.append((g, my_val))
            if their_val < -0.3:
                they_need.append((g, their_val))
            if their_val > 0.3:
                they_give.append((g, their_val))

        # Compute complementarity score
        # High score = they have what I need AND I have what they need
        complement_score = 0.0
        suggested_receive = []
        suggested_send = []

        for g_need, my_deficit in i_need:
            col = f"{g_need}_Value"
            their_surplus = other_z[col]
            if their_surplus > 0.2:
                complement_score += abs(my_deficit) * their_surplus
                suggested_receive.append((g_need, round(float(my_deficit), 2), round(float(their_surplus), 2)))

        for g_give, my_surplus in i_give:
            col = f"{g_give}_Value"
            their_deficit = other_z[col]
            if their_deficit < -0.2:
                complement_score += my_surplus * abs(their_deficit)
                suggested_send.append((g_give, round(float(my_surplus), 2), round(float(their_deficit), 2)))

        # Category-level complementarity
        # Track specific categories where we complement each other so the UI
        # can suggest players based on category needs even when there are no
        # positional gaps.
        cat_complement = 0.0
        cats_i_need = []   # (cat, my_z, their_z) – cats where I'm weak & they're strong
        cats_i_offer = []  # (cat, my_z, their_z) – cats where I'm strong & they're weak
        if my_team in team_strengths.index and other_team in team_strengths.index:
            my_cat_z = team_strengths.loc[my_team]
            their_cat_z = team_strengths.loc[other_team]
            for c in cats:
                if c in my_cat_z.index and c in their_cat_z.index:
                    mz = float(my_cat_z[c])
                    tz = float(their_cat_z[c])
                    # I'm weak in this cat and they're strong → good
                    if mz < -0.3 and tz > 0.3:
                        cat_complement += abs(mz) * tz * 0.5
                        cats_i_need.append((c, round(mz, 2), round(tz, 2)))
                    # I'm strong and they're weak → also good (mutual benefit)
                    if mz > 0.3 and tz < -0.3:
                        cat_complement += mz * abs(tz) * 0.5
                        cats_i_offer.append((c, round(mz, 2), round(tz, 2)))

        total_score = complement_score + cat_complement

        if total_score > 0:
            results.append({
                "Partner": other_team,
                "Match_Score": round(total_score, 2),
                "Positional_Fit": round(complement_score, 2),
                "Category_Fit": round(cat_complement, 2),
                "I_Receive_Positions": suggested_receive,
                "I_Send_Positions": suggested_send,
                "Cats_I_Need": cats_i_need,
                "Cats_I_Offer": cats_i_offer,
            })

    results.sort(key=lambda x: -x["Match_Score"])
    return results[:top_n]


# Map stat categories to the projection columns that drive them, plus whether
# higher is better.  This lets us rank a partner's players by how much they
# would help the categories we're weak in.
_CAT_TO_STAT_COLS: Dict[str, List[str]] = {
    "R": ["R"],
    "RBI": ["RBI"],
    "HR": ["HR"],
    "TB": ["TB"],
    "SB": ["SB"],
    "AVG": ["AVG", "H", "AB"],
    "OBP": ["OBP"],
    "OPS": ["OPS"],
    "SLG": ["SLG"],
    "H": ["H"],
    "BB": ["BB"],
    "W": ["W"],
    "K": ["K", "SO"],
    "QS": ["QS"],
    "ERA": ["ERA"],
    "WHIP": ["WHIP"],
    "SV": ["SV"],
    "HLD": ["HLD", "HD"],
    "SVHD": ["SVHD", "SV", "HLD", "HD"],
    "IP": ["IP"],
    "GS": ["GS"],
    "K/BB": ["K/BB"],
}

# Categories where *lower* is better
_LOWER_IS_BETTER = {"ERA", "WHIP"}


def rank_players_by_category_need(
    partner_players: pd.DataFrame,
    cats_needed: List[str],
    top_n: int = 8,
) -> pd.DataFrame:
    """Score & rank a partner's players by how much they help the given categories.

    For each player we compute a simple "category help" score:
      For each needed category, z-score the player's relevant stat among the
      partner's roster, then sum.  Higher = this player helps the most in
      the categories you need.
    """
    if partner_players.empty or not cats_needed:
        return pd.DataFrame()

    df = partner_players.copy()
    df["_cat_help"] = 0.0

    for cat in cats_needed:
        stat_cols = _CAT_TO_STAT_COLS.get(cat, [cat])
        # Find the first available column
        col = None
        for sc in stat_cols:
            if sc in df.columns:
                col = sc
                break
        if col is None:
            continue

        vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        mu = vals.mean()
        sd = vals.std()
        if sd > 0:
            z = (vals - mu) / sd
        else:
            z = 0.0

        # For ERA/WHIP, lower is better so flip the sign
        if cat in _LOWER_IS_BETTER:
            z = -z

        df["_cat_help"] += z

    return df.sort_values("_cat_help", ascending=False).head(top_n)


def apply_lineup_overrides(
    players: pd.DataFrame,
    overrides: Dict[tuple, str],
    team: str,
    available_slots: List[str],
    value_col: str = "Value",
    bench_slots: int = 10,
) -> Tuple[pd.DataFrame, float]:
    """Re-optimize lineup but with certain player→slot assignments pinned.

    overrides: {(team_name, player_name): forced_slot}
    Returns the same format as optimize_lineup.
    """
    players = players.copy().reset_index(drop=True)
    players["Value"] = _num_col(players, value_col, default=0.0)

    # Separate pinned players from free players
    pinned_indices = []
    pinned_slots_used = []
    for i, row in players.iterrows():
        key = (team, str(row.get("Player", "")))
        if key in overrides:
            forced_slot = overrides[key]
            players.loc[i, "Assigned_Slot"] = forced_slot
            pinned_indices.append(i)
            if forced_slot in available_slots:
                pinned_slots_used.append(forced_slot)

    # Remaining slots after pinning
    remaining_slots = list(available_slots)
    for s in pinned_slots_used:
        if s in remaining_slots:
            remaining_slots.remove(s)

    # Free players to optimize
    free_mask = ~players.index.isin(pinned_indices)
    free_players = players[free_mask].copy().reset_index(drop=True)

    if len(free_players) > 0 and len(remaining_slots) > 0:
        optimized, _ = optimize_lineup(free_players, "Value", remaining_slots, bench_slots)
        # Write back
        free_idx_map = players[free_mask].index.tolist()
        for new_i, orig_i in enumerate(free_idx_map):
            if new_i < len(optimized):
                players.loc[orig_i, "Assigned_Slot"] = optimized.iloc[new_i]["Assigned_Slot"]
    elif len(free_players) > 0:
        # No remaining slots — all go to bench
        for orig_i in players[free_mask].index:
            players.loc[orig_i, "Assigned_Slot"] = "BENCH"

    # Mark any pinned-to-BENCH players
    for i in pinned_indices:
        key = (team, str(players.loc[i, "Player"]))
        if key in overrides:
            players.loc[i, "Assigned_Slot"] = overrides[key]

    in_lineup = players["Assigned_Slot"].isin(available_slots)
    lineup_value = float(players.loc[in_lineup, "Value"].sum())

    return players.sort_values(by=["Assigned_Slot", "Value"], ascending=[True, False]), lineup_value


# -----------------------------
# UI Components
# -----------------------------
def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>⚾ ESPN Dynasty League Projections</h1>
        <p>Powered by ESPN Fantasy API & FanGraphs Projections</p>
    </div>
    """, unsafe_allow_html=True)


def format_standings_table(df: pd.DataFrame, hit_cats: List[str], pit_cats: List[str]) -> pd.DataFrame:
    display_df = df.copy()
    display_df.insert(0, "Rank", range(1, len(display_df) + 1))
    
    for col in hit_cats + pit_cats + ["Team_Score"]:
        if col in display_df.columns:
            if col in ["OBP", "SLG", "OPS", "AVG", "ERA", "WHIP", "K/BB"]:
                display_df[col] = display_df[col].round(3)
            elif col == "Team_Score":
                display_df[col] = display_df[col].round(2)
            else:
                display_df[col] = display_df[col].round(0).astype(int)
    
    return display_df


# -----------------------------
# Main Application
# -----------------------------
def main():
    render_header()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🏆 ESPN LEAGUE")
        
        # League ID input
        league_id = st.text_input(
            "League ID",
            value=st.session_state.get("espn_league_id", ""),
            placeholder="e.g., 12345678",
            help="Find this in your ESPN league URL"
        )
        
        if league_id:
            st.session_state["espn_league_id"] = league_id
        
        # Season year
        current_year = datetime.now().year
        season_year = st.number_input(
            "Season Year",
            min_value=2018,
            max_value=current_year + 1,
            value=current_year,
            step=1
        )
        
        st.divider()
        
        # Authentication
        st.markdown("### 🔐 AUTHENTICATION")
        st.caption("Required for private leagues")
        
        with st.expander("ESPN Cookies", expanded=False):
            espn_s2 = st.text_input(
                "espn_s2 Cookie",
                type="password",
                value=st.session_state.get("espn_s2", ""),
                help="Find in browser Developer Tools → Application → Cookies"
            )
            swid = st.text_input(
                "SWID Cookie",
                value=st.session_state.get("swid", ""),
                help="Usually looks like {XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX}"
            )
            
            if espn_s2:
                st.session_state["espn_s2"] = espn_s2
            if swid:
                st.session_state["swid"] = swid
            
            st.markdown("""
            **How to find cookies:**
            1. Log into ESPN Fantasy
            2. Open Developer Tools (F12)
            3. Go to Application → Cookies
            4. Find `espn_s2` and `SWID`
            """)
        
        refresh_btn = st.button("🔄 Refresh League", use_container_width=True, type="primary")
        
        st.divider()
        
        # Projection system selection
        st.markdown("### 📈 PROJECTIONS")
        
        hitting_proj = st.selectbox(
            "Hitting Projections",
            options=list(HITTING_PROJECTIONS.keys()),
            index=0
        )
        
        pitching_proj = st.selectbox(
            "Pitching Projections",
            options=list(PITCHING_PROJECTIONS.keys()),
            index=0
        )
        
        st.divider()
        
        # Categories
        st.markdown("### 📊 Categories")
        
        with st.expander("⚾ HITTING", expanded=True):
            default_hit_cats = ["R", "RBI", "HR", "TB", "SB", "AVG", "OPS"]
            extra_hit_cats = ["H",  "BB", "SO", "wOBA", "OBP", "SLG"]
            hit_cats = st.multiselect(
                "Hitting categories",
                options=default_hit_cats + extra_hit_cats,
                default=default_hit_cats,
                label_visibility="collapsed"
            )
        
        with st.expander("⚾ PITCHING", expanded=True):
            default_pit_cats = ["W", "K", "QS", "ERA", "WHIP", "SV", "HLD"]
            extra_pit_cats = ["IP", "GS", "BB", "HR", "K/9", "BB/9", "K/BB", "SVHD"]
            pit_cats = st.multiselect(
                "Pitching categories",
                options=default_pit_cats + extra_pit_cats,
                default=default_pit_cats,
                label_visibility="collapsed"
            )
        
        st.divider()
        
        # Roster settings
        st.markdown("### 👥 ROSTER SETTINGS")
        
        with st.expander("Lineup Slots", expanded=False):
            st.markdown("##### Hitters")
            col1, col2 = st.columns(2)
            with col1:
                bat_c = st.number_input("C", min_value=0, value=1, step=1, key="slot_c")
                bat_1b = st.number_input("1B", min_value=0, value=1, step=1, key="slot_1b")
                bat_2b = st.number_input("2B", min_value=0, value=1, step=1, key="slot_2b")
                bat_ss = st.number_input("SS", min_value=0, value=1, step=1, key="slot_ss")
                bat_3b = st.number_input("3B", min_value=0, value=1, step=1, key="slot_3b")
            with col2:
                bat_of = st.number_input("OF", min_value=0, value=3, step=1, key="slot_of")
                bat_lf = st.number_input("LF", min_value=0, value=0, step=1, key="slot_lf")
                bat_cf = st.number_input("CF", min_value=0, value=0, step=1, key="slot_cf")
                bat_rf = st.number_input("RF", min_value=0, value=0, step=1, key="slot_rf")
                bat_util = st.number_input("UTIL", min_value=0, value=2, step=1, key="slot_util")
            
            col1, col2 = st.columns(2)
            with col1:
                bat_mi = st.number_input("MI (2B/SS)", min_value=0, value=0, step=1, key="slot_mi")
            with col2:
                bat_ci = st.number_input("CI (1B/3B)", min_value=0, value=0, step=1, key="slot_ci")
            
            st.markdown("##### Pitchers")
            col1, col2 = st.columns(2)
            with col1:
                pit_sp = st.number_input("SP", min_value=0, value=2, step=1, key="slot_sp")
                pit_rp = st.number_input("RP", min_value=0, value=2, step=1, key="slot_rp")
            with col2:
                pit_p = st.number_input("P", min_value=0, value=5, step=1, key="slot_p")
            
            bench_slots = st.number_input("Bench", min_value=0, value=8, step=1, key="slot_bench")
        
        # Build slot lists
        hitter_slots = (
            ["C"] * bat_c + ["1B"] * bat_1b + ["2B"] * bat_2b +
            ["3B"] * bat_3b + ["SS"] * bat_ss +
            ["OF"] * bat_of + ["LF"] * bat_lf + ["CF"] * bat_cf + ["RF"] * bat_rf +
            ["MI"] * bat_mi + ["CI"] * bat_ci +
            ["UTIL"] * bat_util
        )
        pitcher_slots = ["SP"] * pit_sp + ["RP"] * pit_rp + ["P"] * pit_p
        
        st.divider()
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            match_threshold = st.slider("Name match threshold", 50, 95, 80, 1)
            min_pa = st.number_input("Min PA (hitter eligibility)", value=100, step=25, min_value=0)
            min_ip = st.number_input("Min IP (pitcher eligibility)", value=30.0, step=5.0, min_value=0.0)
            
            st.divider()
            st.markdown("##### Standings Mode")
            include_bench = st.toggle(
                "Include bench in category totals",
                value=False,
                help=(
                    "When ON, category totals use all rostered players — not just "
                    "the optimized starting lineup. Useful for counting stats "
                    "like HLD and SV where relievers on your bench still contribute."
                ),
                key="include_bench_toggle",
            )
            
            st.divider()
            st.markdown("##### Category Weights")
            weights: Dict[str, float] = {}
            for c in hit_cats + pit_cats:
                weights[c] = st.number_input(f"{c} weight", value=1.0, step=0.5, key=f"weight_{c}")
    
    # Main content area
    if not league_id:
        st.info("👈 Enter your ESPN League ID in the sidebar to get started.")
        
        st.markdown("""
        ### How to find your ESPN League ID
        
        1. Go to your ESPN Fantasy Baseball league page
        2. Look at the URL: `https://fantasy.espn.com/baseball/league?leagueId=XXXXXXXX`
        3. Copy the number after `leagueId=`
        
        ### For Private Leagues
        
        You'll also need to provide authentication cookies:
        1. Log into ESPN in your browser
        2. Open Developer Tools (F12)
        3. Go to Application → Cookies → espn.com
        4. Copy the values for `espn_s2` and `SWID`
        
        ### Features
        
        - 📊 **Automatic roster import** from ESPN API
        - 🎯 **Multiple projection systems** (Steamer, THE BAT X, ATC, and more)
        - 📈 **Team standings projection** based on category z-scores
        - ⚙️ **Lineup optimization** with manual slot overrides
        - 🔀 **Trade matchmaker** — find complementary trade partners by positional value
        """)
        st.stop()
    
    # Load data
    espn_s2 = st.session_state.get("espn_s2", "")
    swid = st.session_state.get("swid", "")
    
    with st.spinner("Loading league data from ESPN..."):
        # Fetch league info
        league_info, league_err = fetch_espn_league_info(league_id, season_year, espn_s2, swid)
        if league_err:
            st.error(f"❌ Could not load league: {league_err}")
            st.stop()
        
        league_name = league_info.get("leagueName", "ESPN League")
        
        # Fetch rosters
        roster, roster_err, raw_debug = fetch_espn_rosters(league_id, season_year, espn_s2, swid)
        if roster_err:
            st.error(f"❌ Could not load rosters: {roster_err}")
            if raw_debug:
                with st.expander("🔧 Raw API Debug"):
                    st.json(raw_debug)
            st.stop()
    
    if roster.empty:
        st.error("No roster data found.")
        st.stop()
    
    # Load projections
    hit_proj_type = HITTING_PROJECTIONS[hitting_proj]
    pit_proj_type = PITCHING_PROJECTIONS[pitching_proj]
    
    with st.spinner(f"Loading {hitting_proj}/{pitching_proj} projections..."):
        bat_df, bat_err = fetch_fangraphs_projections(hit_proj_type, is_pitching=False)
        pit_df, pit_err = fetch_fangraphs_projections(pit_proj_type, is_pitching=True)
    
    if bat_err:
        st.warning(f"⚠️ Batting projection error: {bat_err}")
    if pit_err:
        st.warning(f"⚠️ Pitching projection error: {pit_err}")
    
    if bat_df.empty and pit_df.empty:
        st.error("Could not load projections from FanGraphs.")
        st.stop()
    
    # Build projection index and match
    bat, pit = build_projection_index(bat_df, pit_df)

    # Pre-derive computed stats on the projection DataFrames so ALL
    # downstream consumers (add_projection_stats, two-way handling, etc.) see them.
    if "TB" not in bat.columns and len(bat) > 0:
        _h = pd.to_numeric(bat.get("H", 0), errors="coerce").fillna(0)
        _hr = pd.to_numeric(bat.get("HR", 0), errors="coerce").fillna(0)
        _2b = pd.to_numeric(bat.get("2B", bat.get("X2B", 0)), errors="coerce").fillna(0)
        _3b = pd.to_numeric(bat.get("3B", bat.get("X3B", 0)), errors="coerce").fillna(0)
        bat["TB"] = (_h - _2b - _3b - _hr) + 2 * _2b + 3 * _3b + 4 * _hr
    if len(pit) > 0:
        if "K" not in pit.columns and "SO" in pit.columns:
            pit["K"] = pit["SO"]
        if "SVHD" not in pit.columns:
            _sv = pd.to_numeric(pit.get("SV", 0), errors="coerce").fillna(0)
            _hld = pd.to_numeric(pit.get("HLD", pit.get("HD", 0)), errors="coerce").fillna(0)
            pit["SVHD"] = 2 * _sv + _hld
        if "K/BB" not in pit.columns:
            _pk = pd.to_numeric(pit.get("K", pit.get("SO", 0)), errors="coerce").fillna(0)
            _pbb = pd.to_numeric(pit.get("BB", 0), errors="coerce").fillna(0)
            pit["K/BB"] = np.where(_pbb > 0, _pk / _pbb, np.nan)

    roster_matched = match_roster_to_projections(roster, bat, pit, threshold=match_threshold)
    
    # Apply any manual match overrides from session state
    if "match_overrides" not in st.session_state:
        st.session_state["match_overrides"] = {}
    if "lineup_overrides" not in st.session_state:
        st.session_state["lineup_overrides"] = {}  # {(team, player): slot}
    if st.session_state["match_overrides"]:
        roster_matched = apply_match_overrides(roster_matched, st.session_state["match_overrides"])
    
    # Determine Use_As based on IsPitcher flag
    roster_matched["Use_As"] = roster_matched["IsPitcher"].apply(lambda x: "Pitcher" if x else "Hitter")
    
    # Add projection stats
    joined = add_projection_stats(roster_matched, bat, pit)

    # ------------------------------------------------------------------
    # Two-way player handling (e.g. Shohei Ohtani)
    # A player flagged IsPitcher who ALSO exists in the batting projections
    # gets a duplicate "Hitter" row so the optimizer can slot them into a
    # UTIL/DH spot for their bat while keeping the pitcher copy for SP/P.
    # ------------------------------------------------------------------
    bat_clean_names = set(bat["__CleanName"].dropna().unique()) if "__CleanName" in bat.columns else set()
    two_way_rows = []
    for idx, row in joined.iterrows():
        if row.get("Use_As") != "Pitcher":
            continue
        clean = _clean_name(str(row.get("Player", "")))
        # Check if this pitcher also has batting projections
        if clean not in bat_clean_names:
            # Try the matched name too (in case of fuzzy match)
            matched = str(row.get("Matched_Name", ""))
            if not matched or matched not in bat_clean_names:
                continue
            clean = matched

        # Look up their batting projection row
        bat_match = bat[bat["__CleanName"] == clean]
        if bat_match.empty:
            continue

        fg_bat = bat_match.iloc[0]
        # Only create the hitter copy if they have meaningful PA
        pa = pd.to_numeric(fg_bat.get("PA", 0), errors="coerce")
        if pd.isna(pa) or pa < 50:
            continue

        # Build the hitter duplicate
        hitter_row = row.copy()
        hitter_row["Use_As"] = "Hitter"
        hitter_row["IsPitcher"] = False
        hitter_row["Matched_Type"] = "Hitter"
        hitter_row["_TwoWay"] = True
        # Ensure UTIL/DH eligibility even if their ESPN position string is all pitching
        existing_pos = str(hitter_row.get("Position", ""))
        if "DH" not in existing_pos and "UTIL" not in existing_pos:
            hitter_row["Position"] = existing_pos + ",DH" if existing_pos else "DH"

        # Overlay batting projection stats onto the hitter copy
        bat_name_col = _name_col(bat)
        fg_dict = fg_bat.to_dict()
        if "Team" in fg_dict:
            fg_dict["MLB_Team"] = fg_dict.pop("Team")
        if "Name" in fg_dict:
            fg_dict["FG_Name"] = fg_dict.pop("Name")
        # Only copy stats that aren't already roster metadata
        roster_keys = {"Team", "TeamID", "Player", "ESPN_ID", "Position",
                       "LineupSlot", "IsPitcher", "Use_As", "Matched_Name",
                       "Matched_Type", "Match_Score", "__CleanName",
                       "__CleanMatched", "_TwoWay"}
        for k, v in fg_dict.items():
            if k not in roster_keys:
                hitter_row[k] = v

        two_way_rows.append(hitter_row)

    if two_way_rows:
        two_way_df = pd.DataFrame(two_way_rows)
        joined = pd.concat([joined, two_way_df], ignore_index=True)
        # Mark original pitcher rows so we can identify them later
        if "_TwoWay" not in joined.columns:
            joined["_TwoWay"] = False
        joined["_TwoWay"] = joined["_TwoWay"].fillna(False)
    else:
        joined["_TwoWay"] = False
    
    # Build pools
    hit_pool = joined[joined["Use_As"] == "Hitter"].copy()
    pit_pool = joined[joined["Use_As"] == "Pitcher"].copy()
    
    # Eligibility
    hit_elig = _num_col(hit_pool, "PA", default=0.0) >= float(min_pa)
    pit_elig = _num_col(pit_pool, "IP", default=0.0) >= float(min_ip)
    
    # Z-scores
    hit_pool_z = compute_zscores(hit_pool, hit_cats, invert=set())
    pit_pool_z = compute_zscores(pit_pool, pit_cats, invert={"ERA", "WHIP"})
    
    hit_pool["Value"] = 0.0
    for c in hit_cats:
        hit_pool.loc[hit_elig, "Value"] += hit_pool_z.loc[hit_elig, c + "_z"].fillna(0.0) * float(weights.get(c, 1.0))
    
    pit_pool["Value"] = 0.0
    for c in pit_cats:
        pit_pool.loc[pit_elig, "Value"] += pit_pool_z.loc[pit_elig, c + "_z"].fillna(0.0) * float(weights.get(c, 1.0))
    
    scored = pd.concat([hit_pool, pit_pool], ignore_index=True)
    scored["Value"] = pd.to_numeric(scored["Value"], errors="coerce").fillna(0.0)
    
    # Summary metrics (exclude two-way duplicates from player count)
    teams = sorted(scored["Team"].dropna().unique().tolist())
    unique_players = scored[~scored.get("_TwoWay", False).astype(bool)] if "_TwoWay" in scored.columns else scored
    matched_ok = unique_players["Matched_Name"].astype(str).str.len() > 0
    low_conf = _num_col(unique_players, "Match_Score", default=0.0) < float(match_threshold)
    
    n_two_way = int(scored["_TwoWay"].sum()) if "_TwoWay" in scored.columns else 0

    st.markdown(f"## 📊 {league_name}")
    st.caption(f"Season {season_year} • Projections: {hitting_proj} (Hitting) • {pitching_proj} (Pitching)")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Teams", len(teams))
    col2.metric("Total Players", len(unique_players), delta=f"+{n_two_way} two-way" if n_two_way else None)
    col3.metric("Matched", int(matched_ok.sum()))
    col4.metric("Unmatched", int((~matched_ok | low_conf).sum()))
    
    # Calculate team totals and standings
    team_totals_rows = []       # starters-only totals
    team_totals_full_rows = []  # full-roster totals (starters + bench)
    team_depth_rows = []        # bench depth scores
    lineups: Dict[str, Dict[str, pd.DataFrame]] = {}
    
    for t in teams:
        t_rows = scored[scored["Team"] == t].copy()
        hitters = t_rows[t_rows["Use_As"] == "Hitter"].copy()
        pitchers = t_rows[t_rows["Use_As"] == "Pitcher"].copy()
        
        # Check if any lineup overrides exist for this team
        team_hit_overrides = {k: v for k, v in st.session_state["lineup_overrides"].items() if k[0] == t}
        has_hit_overrides = any(k[1] in hitters["Player"].values for k in team_hit_overrides)
        has_pit_overrides = any(k[1] in pitchers["Player"].values for k in team_hit_overrides)

        if has_hit_overrides and team_hit_overrides:
            hit_assigned, _ = apply_lineup_overrides(hitters, team_hit_overrides, t, hitter_slots, "Value", bench_slots)
        else:
            hit_assigned, _ = optimize_lineup(hitters, "Value", hitter_slots, bench_slots)

        if has_pit_overrides and team_hit_overrides:
            pit_assigned, _ = apply_lineup_overrides(pitchers, team_hit_overrides, t, pitcher_slots, "Value", bench_slots)
        else:
            pit_assigned, _ = optimize_lineup(pitchers, "Value", pitcher_slots, bench_slots)
        
        # Filter starters vs bench
        if "Assigned_Slot" in hit_assigned.columns and len(hit_assigned) > 0:
            hit_starters = hit_assigned[hit_assigned["Assigned_Slot"].isin(hitter_slots)].copy()
            hit_bench = hit_assigned[~hit_assigned["Assigned_Slot"].isin(hitter_slots)].copy()
        else:
            hit_starters = pd.DataFrame()
            hit_bench = pd.DataFrame()
        
        if "Assigned_Slot" in pit_assigned.columns and len(pit_assigned) > 0:
            pit_starters = pit_assigned[pit_assigned["Assigned_Slot"].isin(pitcher_slots)].copy()
            pit_bench = pit_assigned[~pit_assigned["Assigned_Slot"].isin(pitcher_slots)].copy()
        else:
            pit_starters = pd.DataFrame()
            pit_bench = pd.DataFrame()
        
        # Starters-only totals
        starter_totals = team_totals_from_lineup(hit_starters, pit_starters, hit_cats, pit_cats)
        starter_totals["Team"] = t
        team_totals_rows.append(starter_totals)
        
        # Full-roster totals (starters + bench)
        full_totals = team_totals_from_lineup(hit_assigned, pit_assigned, hit_cats, pit_cats)
        full_totals["Team"] = t
        team_totals_full_rows.append(full_totals)
        
        # --- Depth score ---
        hit_bench_val = float(hit_bench["Value"].sum()) if not hit_bench.empty and "Value" in hit_bench.columns else 0.0
        pit_bench_val = float(pit_bench["Value"].sum()) if not pit_bench.empty and "Value" in pit_bench.columns else 0.0
        starter_val = float(hit_starters["Value"].sum()) if not hit_starters.empty and "Value" in hit_starters.columns else 0.0
        starter_val += float(pit_starters["Value"].sum()) if not pit_starters.empty and "Value" in pit_starters.columns else 0.0
        bench_total = hit_bench_val + pit_bench_val
        n_bench = (len(hit_bench) if not hit_bench.empty else 0) + (len(pit_bench) if not pit_bench.empty else 0)
        
        team_depth_rows.append({
            "Team": t,
            "Starter_Value": round(starter_val, 2),
            "Bench_Value": round(bench_total, 2),
            "Total_Value": round(starter_val + bench_total, 2),
            "Bench_Hit_Value": round(hit_bench_val, 2),
            "Bench_Pit_Value": round(pit_bench_val, 2),
            "Bench_Count": n_bench,
            "Bench_Avg": round(bench_total / n_bench, 2) if n_bench > 0 else 0.0,
        })
        
        lineups[t] = {
            "hitters": hit_assigned,
            "pitchers": pit_assigned,
        }
    
    # Build both totals DataFrames
    team_totals_starters = pd.DataFrame(team_totals_rows).set_index("Team")
    team_totals_full = pd.DataFrame(team_totals_full_rows).set_index("Team")
    depth_df = pd.DataFrame(team_depth_rows).set_index("Team")
    
    # Choose which totals to use based on toggle
    team_totals = team_totals_full if include_bench else team_totals_starters
    
    cats = hit_cats + pit_cats
    team_scores = team_scores_from_totals(team_totals[cats], cats, weights)
    team_scores = team_scores.reset_index().rename(columns={"index": "Team"})
    team_scores = team_scores.sort_values("Team_Score", ascending=False).reset_index(drop=True)
    
    # Depth z-scores across the league
    bench_val_mu = depth_df["Bench_Value"].mean()
    bench_val_sd = depth_df["Bench_Value"].std()
    if bench_val_sd > 0:
        depth_df["Depth_Score"] = ((depth_df["Bench_Value"] - bench_val_mu) / bench_val_sd).round(2)
    else:
        depth_df["Depth_Score"] = 0.0
    depth_df = depth_df.sort_values("Bench_Value", ascending=False)
    
    # Calculate team strengths for trade matching
    team_strengths = calculate_team_category_strength(team_totals, cats)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 League Standings",
        "👥 Team Rosters",
        "🔀 Trade Matchmaker",
        "🔗 Match Overrides",
        "🔍 Player Search",
        "🆓 Free Agents"
    ])
    
    with tab1:
        st.markdown("### Projected Rest-of-Season Standings")
        if include_bench:
            st.caption("📦 **Full Roster mode** — totals include all rostered players (starters + bench)")
        else:
            st.caption("Based on optimized lineups • Roto scoring: 1st place = max points")
        
        # Roto standings
        n_teams = len(team_totals)
        rank_df = team_totals[cats].copy()
        
        for c in cats:
            ascending = c in {"ERA", "WHIP"}
            ranks = rank_df[c].rank(ascending=ascending, na_option='bottom')
            roto_points = (n_teams + 1 - ranks).fillna(1).astype(int)
            rank_df[c] = roto_points
        
        hit_cols = [c for c in hit_cats if c in rank_df.columns]
        pit_cols = [c for c in pit_cats if c in rank_df.columns]

        rank_df["Hitting_Points"] = rank_df[hit_cols].sum(axis=1) if hit_cols else 0
        rank_df["Pitching_Points"] = rank_df[pit_cols].sum(axis=1) if pit_cols else 0

        # Existing total
        rank_df["Total_Points"] = rank_df[cats].sum(axis=1)

        # Put the split totals right before Total_Points
        total_idx = rank_df.columns.get_loc("Total_Points")
        rank_df.insert(total_idx, "Pitching_Points", rank_df.pop("Pitching_Points"))
        rank_df.insert(total_idx, "Hitting_Points", rank_df.pop("Hitting_Points"))
        
        rank_df["Total_Points"] = rank_df[cats].sum(axis=1)
        rank_df = rank_df.sort_values("Total_Points", ascending=False)
        rank_df.insert(0, "Rank", range(1, len(rank_df) + 1))
        
        def highlight_standings_rank(val):
            if val == 1:
                return 'background-color: #ffd700; font-weight: bold'
            elif val == 2:
                return 'background-color: #c0c0c0; font-weight: bold'
            elif val == 3:
                return 'background-color: #cd7f32; color: white; font-weight: bold'
            return ''
        
        def highlight_cat_points(val):
            try:
                v = int(val)
                if v >= n_teams - 2:
                    return 'background-color: #d4edda; color: #155724'
                elif v >= (n_teams // 2):
                    return 'background-color: #fff3cd; color: #856404'
                elif v <= 3:
                    return 'background-color: #f8d7da; color: #721c24'
            except:
                pass
            return ''
        
        display_cols = ["Rank"] + cats + ["Hitting_Points", "Pitching_Points", "Total_Points"]

        
        st.dataframe(
            rank_df[display_cols].style
                .applymap(highlight_standings_rank, subset=["Rank"])
                .applymap(highlight_cat_points, subset=cats),
            use_container_width=True,
            height=750
        )
        
        # --- Depth Score Section ---
        with st.expander("🪑 Bench Depth Rankings", expanded=False):
            st.caption(
                "Measures the quality of each team's bench. "
                "**Bench Value** = sum of z-score weighted player values on the bench. "
                "**Depth Score** = standard deviations above/below the league average bench."
            )
            
            depth_display = depth_df[["Starter_Value", "Bench_Value", "Total_Value",
                                      "Bench_Hit_Value", "Bench_Pit_Value",
                                      "Bench_Count", "Bench_Avg", "Depth_Score"]].copy()
            depth_display.insert(0, "Rank", range(1, len(depth_display) + 1))
            
            def _color_depth(val):
                try:
                    v = float(val)
                    if v >= 1.0:
                        return "background-color: #d4edda; color: #155724"
                    elif v >= 0.3:
                        return "background-color: #e8f5e9; color: #2e7d32"
                    elif v <= -1.0:
                        return "background-color: #f8d7da; color: #721c24"
                    elif v <= -0.3:
                        return "background-color: #fce4ec; color: #c62828"
                except:
                    pass
                return ""
            
            st.dataframe(
                depth_display.round(2).style.applymap(
                    _color_depth, subset=["Depth_Score"]
                ),
                use_container_width=True,
                height=min(500, 50 + 35 * len(teams)),
            )
        
        with st.expander("📊 Raw Category Totals"):
            display_df = format_standings_table(team_scores, hit_cats, pit_cats)
            st.dataframe(display_df, use_container_width=True, height=750)
    
    with tab2:
        st.markdown("### Team Roster Breakdown")
        
        team_pick = st.selectbox("Select Team", teams, key="team_select")
        
        if team_pick:
            team_data = lineups[team_pick]
            
            # --- Lineup Override Controls ---
            with st.expander("⚙️ Manual Lineup Slot Overrides", expanded=False):
                st.caption(
                    "Pin a player to a specific slot (or BENCH). The optimizer will fill "
                    "remaining slots around your pinned choices."
                )

                all_team_players = pd.concat(
                    [team_data["hitters"], team_data["pitchers"]], ignore_index=True
                )
                override_player = st.selectbox(
                    "Player to override",
                    ["(select)"] + all_team_players["Player"].tolist(),
                    key="lineup_override_player",
                )

                if override_player != "(select)":
                    player_row = all_team_players[
                        all_team_players["Player"] == override_player
                    ].iloc[0]
                    is_pit = bool(player_row.get("IsPitcher", False))
                    player_positions = _split_positions(str(player_row.get("Position", "")))

                    # Build eligible slot options
                    eligible_slots = ["BENCH"]
                    if is_pit:
                        for s in pitcher_slots:
                            if s not in eligible_slots:
                                eligible_slots.append(s)
                        # Also allow P generic if pitcher
                        if "P" not in eligible_slots and any(
                            p in ["SP", "RP", "P"] for p in player_positions
                        ):
                            eligible_slots.append("P")
                    else:
                        for s in hitter_slots:
                            if s not in eligible_slots:
                                eligible_slots.append(s)

                    current_override = st.session_state["lineup_overrides"].get(
                        (team_pick, override_player), None
                    )
                    current_assigned = str(player_row.get("Assigned_Slot", "BENCH"))

                    default_idx = 0
                    display_val = current_override or current_assigned
                    if display_val in eligible_slots:
                        default_idx = eligible_slots.index(display_val)

                    new_slot = st.selectbox(
                        f"Assign **{override_player}** to slot:",
                        eligible_slots,
                        index=default_idx,
                        key="lineup_override_slot",
                    )

                    ocol1, ocol2 = st.columns(2)
                    with ocol1:
                        if st.button("📌 Pin Override", key="pin_lineup_override"):
                            st.session_state["lineup_overrides"][
                                (team_pick, override_player)
                            ] = new_slot
                            st.rerun()
                    with ocol2:
                        if (team_pick, override_player) in st.session_state[
                            "lineup_overrides"
                        ]:
                            if st.button(
                                "🗑️ Remove Override", key="remove_lineup_override"
                            ):
                                del st.session_state["lineup_overrides"][
                                    (team_pick, override_player)
                                ]
                                st.rerun()

                # Show active lineup overrides
                active_overrides = {
                    k: v
                    for k, v in st.session_state["lineup_overrides"].items()
                    if k[0] == team_pick
                }
                if active_overrides:
                    st.markdown(f"**Active overrides for {team_pick}:**")
                    for (t, p), s in active_overrides.items():
                        st.text(f"  📌 {p} → {s}")
                    if st.button(
                        "🗑️ Clear All Lineup Overrides for This Team",
                        key="clear_team_lineup_overrides",
                    ):
                        st.session_state["lineup_overrides"] = {
                            k: v
                            for k, v in st.session_state["lineup_overrides"].items()
                            if k[0] != team_pick
                        }
                        st.rerun()

            # --- Display Roster ---
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### ⚾ Hitters")
                hitters_df = team_data["hitters"].copy()
                
                display_cols = ["Assigned_Slot", "Player", "Position", "Matched_Name", "Match_Score", "Value"]
                for c in hit_cats:
                    if c in hitters_df.columns:
                        display_cols.append(c)
                
                display_cols = [c for c in display_cols if c in hitters_df.columns]
                
                starters = hitters_df[hitters_df["Assigned_Slot"].isin(hitter_slots)] if "Assigned_Slot" in hitters_df.columns else pd.DataFrame()
                bench = hitters_df[~hitters_df["Assigned_Slot"].isin(hitter_slots)] if "Assigned_Slot" in hitters_df.columns else pd.DataFrame()

                # Highlight pinned players
                pinned_players = {
                    k[1] for k, v in st.session_state["lineup_overrides"].items() if k[0] == team_pick
                }
                two_way_players = set(
                    hitters_df.loc[hitters_df.get("_TwoWay", pd.Series(False, index=hitters_df.index)).astype(bool), "Player"].tolist()
                ) if "_TwoWay" in hitters_df.columns else set()

                st.markdown("**Starting Lineup**")
                if not starters.empty:
                    def _highlight_pinned(row):
                        player = row.get("Player", "")
                        if player in pinned_players:
                            return ["background-color: #fff3cd"] * len(row)
                        if player in two_way_players:
                            return ["background-color: #d1ecf1"] * len(row)
                        return [""] * len(row)
                    st.dataframe(
                        starters[display_cols].round(3).style.apply(_highlight_pinned, axis=1),
                        use_container_width=True, height=350,
                    )
                
                st.markdown("**Bench**")
                if not bench.empty:
                    st.dataframe(
                        bench[display_cols].round(3).style.apply(_highlight_pinned, axis=1),
                        use_container_width=True, height=250,
                    )
            
            with col2:
                st.markdown("#### 🎯 Pitchers")
                pitchers_df = team_data["pitchers"].copy()
                
                display_cols = ["Assigned_Slot", "Player", "Position", "Matched_Name", "Match_Score", "Value"]
                for c in pit_cats:
                    if c in pitchers_df.columns:
                        display_cols.append(c)
                
                display_cols = [c for c in display_cols if c in pitchers_df.columns]
                
                starters = pitchers_df[pitchers_df["Assigned_Slot"].isin(pitcher_slots)] if "Assigned_Slot" in pitchers_df.columns else pd.DataFrame()
                bench = pitchers_df[~pitchers_df["Assigned_Slot"].isin(pitcher_slots)] if "Assigned_Slot" in pitchers_df.columns else pd.DataFrame()
                
                st.markdown("**Starting Rotation**")
                if not starters.empty:
                    st.dataframe(
                        starters[display_cols].round(3).style.apply(_highlight_pinned, axis=1),
                        use_container_width=True, height=350,
                    )
                
                st.markdown("**Bench**")
                if not bench.empty:
                    st.dataframe(
                        bench[display_cols].round(3).style.apply(_highlight_pinned, axis=1),
                        use_container_width=True, height=250,
                    )

            legends = []
            if pinned_players:
                legends.append("📌 Yellow = manually overridden slot")
            if two_way_players:
                legends.append("⚾🎯 Blue = two-way player (hitting copy)")
            if legends:
                st.caption(" · ".join(legends))

    with tab3:
        st.markdown("### 🔀 Trade Matchmaker")
        st.caption(
            "Analyzes each team's total positional value (starters + bench) and category "
            "strengths to find complementary trade partners."
        )

        # Compute positional value across the league
        pos_values = compute_positional_value(scored, teams)

        tmcol1, tmcol2 = st.columns([1, 1])
        with tmcol1:
            my_team = st.selectbox("Your Team", teams, key="trade_my_team")
        with tmcol2:
            top_n_partners = st.slider("Top N partners", 3, len(teams) - 1, min(5, len(teams) - 1), key="trade_top_n")

        # --- League-wide Positional Value Heatmap ---
        with st.expander("📊 League-Wide Positional Value", expanded=False):
            groups = list(POSITION_GROUPS.keys())
            val_cols = [f"{g}_Value" for g in groups]
            display_pos = pos_values[val_cols].copy()
            display_pos.columns = groups

            # Z-score for coloring
            pos_z_display = display_pos.copy()
            for c in groups:
                mu = pos_z_display[c].mean()
                sd = pos_z_display[c].std()
                if sd > 0:
                    pos_z_display[c] = (pos_z_display[c] - mu) / sd
                else:
                    pos_z_display[c] = 0.0

            def _color_pos_z(val):
                try:
                    v = float(val)
                    if v >= 1.0:
                        return "background-color: #d4edda; color: #155724"
                    elif v >= 0.3:
                        return "background-color: #e8f5e9; color: #2e7d32"
                    elif v <= -1.0:
                        return "background-color: #f8d7da; color: #721c24"
                    elif v <= -0.3:
                        return "background-color: #fce4ec; color: #c62828"
                except:
                    pass
                return ""

            st.dataframe(
                display_pos.round(1).style.applymap(_color_pos_z, subset=groups),
                use_container_width=True,
                height=min(400, 50 + 35 * len(teams)),
            )
            st.caption("Green = above-average positional depth · Red = below-average")

        # --- My Team's Positional Profile ---
        st.markdown(f"#### {my_team}'s Positional Profile")

        my_pos = pos_values.loc[my_team]
        groups = list(POSITION_GROUPS.keys())

        profile_cols = st.columns(len(groups))
        for i, g in enumerate(groups):
            val = my_pos[f"{g}_Value"]
            count = int(my_pos[f"{g}_Count"])
            # Z-score relative to league
            col_vals = pos_values[f"{g}_Value"]
            mu = col_vals.mean()
            sd = col_vals.std()
            z = (val - mu) / sd if sd > 0 else 0.0

            emoji = "🟢" if z > 0.5 else ("🔴" if z < -0.5 else "🟡")
            with profile_cols[i]:
                st.metric(
                    label=f"{emoji} {g}",
                    value=f"{val:.1f}",
                    delta=f"{z:+.1f}σ ({count} players)",
                )

        # --- Trade Partners ---
        st.markdown("#### Best Trade Partners")

        matches = find_trade_matches(pos_values, team_strengths, my_team, cats, top_n=top_n_partners)

        if not matches:
            st.info("No strong complementary trade partners found. Your roster may be well-balanced!")
        else:
            for rank, m in enumerate(matches, 1):
                partner = m["Partner"]
                score = m["Match_Score"]

                # Build a compact summary
                receive_str = ", ".join(
                    f"**{g}** (you: {my_z:+.1f}σ, them: {their_z:+.1f}σ)"
                    for g, my_z, their_z in m["I_Receive_Positions"]
                )
                send_str = ", ".join(
                    f"**{g}** (you: {my_z:+.1f}σ, them: {their_z:+.1f}σ)"
                    for g, my_z, their_z in m["I_Send_Positions"]
                )
                cats_need_str = ", ".join(
                    f"**{c}** (you: {mz:+.1f}σ, them: {tz:+.1f}σ)"
                    for c, mz, tz in m.get("Cats_I_Need", [])
                )
                cats_offer_str = ", ".join(
                    f"**{c}** (you: {mz:+.1f}σ, them: {tz:+.1f}σ)"
                    for c, mz, tz in m.get("Cats_I_Offer", [])
                )

                with st.container():
                    st.markdown(f"---")
                    hcol1, hcol2, hcol3 = st.columns([2, 1, 1])
                    with hcol1:
                        st.markdown(f"##### #{rank} — {partner}")
                    with hcol2:
                        st.metric("Trade Match Score", f"{score:.1f}")
                    with hcol3:
                        st.metric(
                            "Pos / Cat Split",
                            f"{m['Positional_Fit']:.1f} / {m['Category_Fit']:.1f}",
                        )

                    if receive_str:
                        st.markdown(f"🟢 **Target positions from {partner}:** {receive_str}")
                    if send_str:
                        st.markdown(f"🔵 **Offer positions to {partner}:** {send_str}")
                    if cats_need_str:
                        st.markdown(f"📈 **Categories to target:** {cats_need_str}")
                    if cats_offer_str:
                        st.markdown(f"📉 **Categories you can help them in:** {cats_offer_str}")

                    # Show specific players they could trade
                    with st.expander(f"👀 View {partner}'s Tradeable Assets"):
                        partner_rows = scored[scored["Team"] == partner].copy()
                        receive_groups = [g for g, _, _ in m["I_Receive_Positions"]]
                        cats_i_need_list = [c for c, _, _ in m.get("Cats_I_Need", [])]

                        shown_something = False

                        # Section 1: Positional targets (if any)
                        if receive_groups:
                            st.markdown("##### By Position Need")
                            for g in receive_groups:
                                grp_pos = POSITION_GROUPS[g]
                                candidates = partner_rows[
                                    partner_rows["Position"].apply(
                                        lambda p: any(
                                            pos in grp_pos for pos in _split_positions(str(p))
                                        )
                                    )
                                ].sort_values("Value", ascending=False)

                                if not candidates.empty:
                                    st.markdown(f"**{g}** players:")
                                    disp = ["Player", "Position", "Value"]
                                    for stat in ["PA", "HR", "SB", "AVG", "OPS", "IP", "K", "ERA", "WHIP", "W", "SV"]:
                                        if stat in candidates.columns:
                                            disp.append(stat)
                                    disp = [c for c in disp if c in candidates.columns]
                                    st.dataframe(
                                        candidates[disp].head(5).round(3),
                                        use_container_width=True,
                                        hide_index=True,
                                    )
                                    shown_something = True

                        # Section 2: Category-based targets (always show when
                        # we have category needs — this is the key fallback)
                        if cats_i_need_list:
                            st.markdown("##### By Category Need")
                            st.caption(
                                f"Your team is weak in {', '.join(cats_i_need_list)}. "
                                f"These {partner} players would help the most in those categories."
                            )

                            # Split into hitters and pitchers for clarity
                            hit_cats_needed = [c for c in cats_i_need_list if c in hit_cats]
                            pit_cats_needed = [c for c in cats_i_need_list if c in pit_cats]

                            if hit_cats_needed:
                                partner_hitters = partner_rows[partner_rows["Use_As"] == "Hitter"]
                                if not partner_hitters.empty:
                                    top_h = rank_players_by_category_need(partner_hitters, hit_cats_needed, top_n=6)
                                    if not top_h.empty:
                                        st.markdown(f"**Hitters** (helps: {', '.join(hit_cats_needed)}):")
                                        disp = ["Player", "Position", "Value"]
                                        for stat in hit_cats_needed + ["PA", "HR", "SB", "AVG", "OPS", "R", "RBI", "TB"]:
                                            if stat in top_h.columns and stat not in disp:
                                                disp.append(stat)
                                        disp = [c for c in disp if c in top_h.columns]
                                        st.dataframe(
                                            top_h[disp].round(3),
                                            use_container_width=True,
                                            hide_index=True,
                                        )
                                        shown_something = True

                            if pit_cats_needed:
                                partner_pitchers = partner_rows[partner_rows["Use_As"] == "Pitcher"]
                                if not partner_pitchers.empty:
                                    top_p = rank_players_by_category_need(partner_pitchers, pit_cats_needed, top_n=6)
                                    if not top_p.empty:
                                        st.markdown(f"**Pitchers** (helps: {', '.join(pit_cats_needed)}):")
                                        disp = ["Player", "Position", "Value"]
                                        for stat in pit_cats_needed + ["IP", "K", "ERA", "WHIP", "W", "SV", "QS"]:
                                            if stat in top_p.columns and stat not in disp:
                                                disp.append(stat)
                                        disp = [c for c in disp if c in top_p.columns]
                                        st.dataframe(
                                            top_p[disp].round(3),
                                            use_container_width=True,
                                            hide_index=True,
                                        )
                                        shown_something = True

                        if not shown_something:
                            # Last resort: just show their best overall players
                            st.markdown("##### Best Overall Players")
                            st.caption("No specific positional or category gaps identified — showing their highest-value players.")
                            best = partner_rows.sort_values("Value", ascending=False).head(8)
                            disp = ["Player", "Position", "Use_As", "Value"]
                            for stat in ["PA", "HR", "SB", "AVG", "OPS", "IP", "K", "ERA", "WHIP", "W", "SV"]:
                                if stat in best.columns:
                                    disp.append(stat)
                            disp = [c for c in disp if c in best.columns]
                            st.dataframe(
                                best[disp].round(3),
                                use_container_width=True,
                                hide_index=True,
                            )
    
    with tab4:
        st.markdown("### 🔗 Name Match Overrides")
        st.caption("Fix unmatched or incorrectly matched players by selecting the correct projection")

        # Filter to players that need attention: unmatched or low-confidence
        needs_attention = scored[
            (scored["Matched_Name"].astype(str).str.len() == 0) |
            (_num_col(scored, "Match_Score", default=0.0) < 100)
        ].copy()
        needs_attention = needs_attention.sort_values("Match_Score", ascending=True)

        if needs_attention.empty:
            st.success("All players are matched with 100% confidence!")
        else:
            # Team filter for the override list
            override_team = st.selectbox(
                "Filter by team",
                ["All Teams"] + teams,
                key="override_team_filter",
            )
            if override_team != "All Teams":
                needs_attention = needs_attention[needs_attention["Team"] == override_team]

            st.markdown(f"**{len(needs_attention)}** players need review")

            changes_made = False
            for idx, (_, row) in enumerate(needs_attention.iterrows()):
                pname = str(row.get("Player", ""))
                pteam = str(row.get("Team", ""))
                is_pit = bool(row.get("IsPitcher", False))
                current_match = str(row.get("Matched_Name", ""))
                current_score = int(row.get("Match_Score", 0))
                current_type = str(row.get("Matched_Type", ""))

                # Get near matches
                candidates = get_near_matches(pname, is_pit, bat, pit, n=8)

                # Build option list
                options_display = ["— No match —"]
                options_data = [("", "")]  # (clean_name, type)
                for cand_clean, cand_display, cand_score, cand_type in candidates:
                    label = f"{cand_display}  [{cand_score}%]"
                    options_display.append(label)
                    options_data.append((cand_clean, cand_type))

                # Figure out current selection index
                current_idx = 0
                if current_match:
                    for oi, (cn, ct) in enumerate(options_data):
                        if cn == current_match:
                            current_idx = oi
                            break

                # Check if there's already an override in session state
                existing_override = st.session_state["match_overrides"].get(pname)
                if existing_override:
                    for oi, (cn, ct) in enumerate(options_data):
                        if cn == existing_override[0]:
                            current_idx = oi
                            break

                # Render row
                match_icon = "✅" if current_score == 100 else ("🟡" if current_score >= 80 else "❌")
                c1, c2, c3 = st.columns([2.5, 4, 1])
                with c1:
                    st.markdown(
                        f"{match_icon} **{pname}**  \n"
                        f"<small style='color:gray'>{pteam} · {'P' if is_pit else 'H'} · "
                        f"Current: {current_match or 'none'} ({current_score}%)</small>",
                        unsafe_allow_html=True,
                    )
                with c2:
                    sel = st.selectbox(
                        f"Match for {pname}",
                        options=options_display,
                        index=current_idx,
                        key=f"match_override_{pname}_{idx}",
                        label_visibility="collapsed",
                    )
                    sel_idx = options_display.index(sel)
                    chosen_clean, chosen_type = options_data[sel_idx]

                    # Detect if user changed the selection from the auto-match
                    if chosen_clean != current_match or (existing_override and chosen_clean != existing_override[0]):
                        st.session_state["match_overrides"][pname] = (chosen_clean, chosen_type)
                        changes_made = True
                with c3:
                    if pname in st.session_state["match_overrides"]:
                        st.caption("✏️ Override")

            st.divider()
            col_a, col_b = st.columns(2)
            with col_a:
                if changes_made:
                    if st.button("🔄 Apply Changes & Recalculate", type="primary", key="apply_match_overrides"):
                        st.rerun()
            with col_b:
                if st.session_state["match_overrides"]:
                    if st.button("🗑️ Clear All Overrides", key="clear_match_overrides"):
                        st.session_state["match_overrides"] = {}
                        st.rerun()

            if st.session_state["match_overrides"]:
                with st.expander(f"📋 Active Overrides ({len(st.session_state['match_overrides'])})"):
                    for pn, (mn, mt) in st.session_state["match_overrides"].items():
                        st.text(f"{pn}  →  {mn or '(unmatched)'} ({mt})")

    with tab5:
        st.markdown("### Player Search")
        
        search_term = st.text_input("🔍 Search players", placeholder="Enter player name...")
        
        if search_term:
            mask = scored["Player"].str.lower().str.contains(search_term.lower(), na=False)
            results = scored[mask].copy()
        else:
            results = scored.copy()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            team_filter = st.multiselect("Filter by team", options=["All"] + teams, default=["All"])
        with col2:
            type_filter = st.selectbox("Player type", ["All", "Hitter", "Pitcher"])
        with col3:
            match_filter = st.selectbox("Match status", ["All", "Matched", "Unmatched"])
        
        if "All" not in team_filter and team_filter:
            results = results[results["Team"].isin(team_filter)]
        
        if type_filter != "All":
            results = results[results["Use_As"] == type_filter]
        
        if match_filter == "Matched":
            results = results[results["Matched_Name"].astype(str).str.len() > 0]
        elif match_filter == "Unmatched":
            results = results[results["Matched_Name"].astype(str).str.len() == 0]
        
        display_cols = ["Team", "Player", "Position", "Matched_Name", "Matched_Type", "Match_Score", "Value"]
        st.dataframe(
            results[display_cols].sort_values("Value", ascending=False).round(2),
            use_container_width=True,
            height=500
        )
    
    with tab6:
        st.markdown("### 🆓 Best Available Free Agents")
        st.caption("Players in projections not currently rostered")
        
        # Get rostered player names
        rostered_names = set(scored["__CleanName"].dropna().unique()) if "__CleanName" in scored.columns else set()
        
        # Combine projections
        all_proj = pd.concat([bat_df, pit_df], ignore_index=True)
        
        # Add clean name
        name_col = _name_col(all_proj)
        all_proj["__CleanName"] = all_proj[name_col].astype(str).apply(_clean_name)
        
        # Filter to free agents
        def is_rostered(clean_name):
            if clean_name in rostered_names:
                return True
            for rostered in rostered_names:
                if fuzz.ratio(clean_name, rostered) >= match_threshold:
                    return True
            return False
        
        all_proj["Is_Rostered"] = all_proj["__CleanName"].apply(is_rostered)
        free_agents = all_proj[~all_proj["Is_Rostered"]].copy()
        
        if free_agents.empty:
            st.info("No free agents found.")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                fa_type = st.selectbox("Player Type", ["All", "Hitters", "Pitchers"], key="fa_type")
            with col2:
                fa_sort = st.selectbox("Sort By", ["PA", "IP", "HR", "SB", "W", "K"], key="fa_sort")
            with col3:
                fa_limit = st.number_input("Show Top N", min_value=10, max_value=500, value=50, step=10, key="fa_limit")
            
            if fa_type == "Hitters":
                fa_display = free_agents[pd.to_numeric(free_agents.get("PA", 0), errors="coerce").fillna(0) > 0].copy()
            elif fa_type == "Pitchers":
                fa_display = free_agents[pd.to_numeric(free_agents.get("IP", 0), errors="coerce").fillna(0) > 0].copy()
            else:
                fa_display = free_agents.copy()
            
            if fa_sort in fa_display.columns:
                fa_display = fa_display.sort_values(fa_sort, ascending=False)
            
            fa_display = fa_display.head(int(fa_limit))
            
            display_cols = [name_col]
            for col in ["PA", "AB", "H", "HR", "R", "RBI", "SB", "OBP", "SLG", "IP", "W", "K", "ERA", "WHIP", "SV"]:
                if col in fa_display.columns:
                    display_cols.append(col)
            
            st.dataframe(
                fa_display[display_cols].round(2),
                use_container_width=True,
                height=600
            )
            
            st.caption(f"Showing {len(fa_display)} of {len(free_agents)} available free agents")
    
    # Debug info
    with st.expander("🔧 Debug Info"):
        st.write("**ESPN League ID:**", league_id)
        st.write("**Season:**", season_year)
        st.write("**Projection Systems:**", f"{hitting_proj} (Hitting), {pitching_proj} (Pitching)")
        st.write("**Roster shape:**", roster.shape)
        st.write("**Teams found:**", teams)
        st.write("**Batting projections shape:**", bat_df.shape)
        st.write("**Pitching projections shape:**", pit_df.shape)
        st.write("**Players per team:**")
        team_counts = roster.groupby("Team").size().to_dict()
        st.write(team_counts)


if __name__ == "__main__":
    main()

import dash
from dash import dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import unicodedata

try:
    from pybaseball import statcast_batter, statcast_pitcher, team_batting, team_pitching, batting_stats
    PYBASEBALL_AVAILABLE = True
except ImportError:
    PYBASEBALL_AVAILABLE = False
    print("PyBaseball not installed. Using cached data only.")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner th {
                position: relative;
            }
            .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner th .column-header--sort {
                position: absolute !important;
                top: 0 !important;
                left: 0 !important;
                right: 0 !important;
                bottom: 0 !important;
                width: 100% !important;
                height: 100% !important;
                padding: 8px !important;
            }
            .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner th .column-header--sort svg {
                position: absolute;
                right: 8px;
                top: 50%;
                transform: translateY(-50%);
            }
            /* Fix dropdown menu options to have dark background and white text */
            .Select-menu-outer {
                background-color: #2d2d2d !important;
            }
            .Select-option {
                background-color: #2d2d2d !important;
                color: #ffffff !important;
            }
            .Select-option:hover {
                background-color: #444444 !important;
                color: #ffffff !important;
            }
            .VirtualizedSelectOption {
                background-color: #2d2d2d !important;
                color: #ffffff !important;
            }
            .VirtualizedSelectFocusedOption {
                background-color: #444444 !important;
                color: #ffffff !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

def generate_arc_points(p1, p2, apex, num_points=100):
    t = np.linspace(0, 1, num_points)
    x = (1 - t)**2 * p1[0] + 2 * (1 - t) * t * apex[0] + t**2 * p2[0]
    y = (1 - t)**2 * p1[1] + 2 * (1 - t) * t * apex[1] + t**2 * p2[1]
    z = (1 - t)**2 * p1[2] + 2 * (1 - t) * t * apex[2] + t**2 * p2[2]
    return x, y, z

def plot_curve(x_start, y_start, z_start, x_end, y_end, z_end, pfx_x, pfx_z, pitch_name):
    t = np.linspace(0, 1, 100)  
    
    if pitch_name in ["Curveball", "Knuckle Curve"]:
        pfx_x = -pfx_x  
        pfx_z = -pfx_z  

    x_curve = x_start + (x_end - x_start) * t + pfx_x * t * (1 - t)
    y_curve = y_start + (y_end - y_start) * t
    z_curve = z_start + (z_end - z_start) * t + pfx_z * t * (1 - t)
    
    return x_curve, y_curve, z_curve

def get_player_hit_data(player_id, year=2025, use_cache=True):
    cache_dir = Path('cache')
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f'player_{player_id}_{year}_hits.pkl'
    
    if use_cache and cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except:
            pass
    
    # fetch from pybaseball
    if PYBASEBALL_AVAILABLE:
        try:
            start_date = f'{year}-04-01'
            end_date = f'{year}-10-31'
            print(f"Fetching Statcast data for player {player_id}...")
            data = statcast_batter(start_date, end_date, player_id)
            
            print(f"  Data type: {type(data)}")
            print(f"  Data is None: {data is None}")
            if data is not None:
                print(f"  Data empty: {data.empty}")
                print(f"  Data shape: {data.shape}")
                print(f"  Columns: {list(data.columns)[:10]}...")  # first 10 cols
            
            if data is not None and not data.empty:
                # check if events column exists
                if 'events' not in data.columns:
                    print(f"  ERROR: No 'events' column found in data!")
                    print(f"  Available columns: {list(data.columns)}")
                    return None
                
                relevant_events = ['single', 'double', 'triple', 'home_run']
                filtered_data = data[data['events'].isin(relevant_events)].copy()
                
                print(f"  Found {len(data)} total at-bats, {len(filtered_data)} hits to cache")
                
                #only keep what we need
                needed_cols = ['events', 'hc_x', 'hc_y', 'launch_angle', 'launch_speed', 
                              'hit_distance_sc', 'des', 'pitcher', 'game_date']
                cols_to_keep = [col for col in needed_cols if col in filtered_data.columns]
                filtered_data = filtered_data[cols_to_keep]
                
                # fix HR coordinates, ground balls are weird so leave those alone
                HOME_X, HOME_Y = 125.42, 198.27
                FEET_PER_UNIT = 2.5
                corrected_count = 0
                
                for idx in filtered_data.index:
                    if filtered_data.loc[idx, 'events'] == 'home_run':
                        if pd.notna(filtered_data.loc[idx, 'hc_x']) and pd.notna(filtered_data.loc[idx, 'hc_y']) and pd.notna(filtered_data.loc[idx, 'hit_distance_sc']):
                            cx = (filtered_data.loc[idx, 'hc_x'] - HOME_X) * FEET_PER_UNIT
                            cy = (HOME_Y - filtered_data.loc[idx, 'hc_y']) * FEET_PER_UNIT
                            ground_dist = np.sqrt(cx**2 + cy**2)
                            actual_dist = filtered_data.loc[idx, 'hit_distance_sc']
                            
                            #correct if error >5%
                            if ground_dist > 0 and actual_dist > 0 and abs(ground_dist - actual_dist) / actual_dist > 0.05:
                                scale_factor = actual_dist / ground_dist
                                corrected_cx = cx * scale_factor
                                corrected_cy = cy * scale_factor
                                
                                filtered_data.loc[idx, 'hc_x'] = (corrected_cx / FEET_PER_UNIT) + HOME_X
                                filtered_data.loc[idx, 'hc_y'] = HOME_Y - (corrected_cy / FEET_PER_UNIT)
                                corrected_count += 1
                
                if corrected_count > 0:
                    print(f"  Corrected {corrected_count} HR coordinate outliers (>5% error)")
                
                print(f"  Caching {len(cols_to_keep)} columns (was 118)")
                
                if not filtered_data.empty:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(filtered_data, f)
                    return filtered_data
        except Exception as e:
            import traceback
            print(f"Error fetching data for player {player_id}: {e}")
            print(f"Full traceback:")
            traceback.print_exc()
    
    return None

def get_player_pitch_data(player_id, year=2025, use_cache=True):
    cache_dir = Path('cache')
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f'pitcher_{player_id}_{year}_pitches.pkl'
    
    if use_cache and cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except:
            pass
    
    if PYBASEBALL_AVAILABLE:
        try:
            start_date = f'{year}-04-01'
            end_date = f'{year}-10-31'
            print(f"Fetching Statcast pitch data for pitcher {player_id}...")
            data = statcast_pitcher(start_date, end_date, player_id)
            
            if data is not None and not data.empty:
                print(f"  Found {len(data)} pitches")
                
                needed_cols = ['pitch_type', 'plate_x', 'plate_z', 'release_speed', 
                              'events', 'description', 'game_date', 'des', 'pfx_x', 'pfx_z']
                cols_to_keep = [col for col in needed_cols if col in data.columns]
                filtered_data = data[cols_to_keep].copy()
                
                print(f"  Caching {len(cols_to_keep)} columns")
                
                with open(cache_file, 'wb') as f:
                    pickle.dump(filtered_data, f)
                return filtered_data
        except Exception as e:
            print(f"Error fetching data for pitcher {player_id}: {e}")
    
    return None

def create_3d_baseball_field():
    fig = go.Figure()
    
    theta = np.linspace(np.radians(45), np.radians(135), 50)
    wall_distance = 350
    
    # center field needs to bulge out a bit
    bulge_amount = 62.5
    normalized = (theta - np.radians(45)) / (np.radians(135) - np.radians(45))
    bulge = bulge_amount * np.sin(normalized * np.pi)
    
    wall_x = (wall_distance + bulge) * np.cos(theta)
    wall_y = (wall_distance + bulge) * np.sin(theta)
    wall_z = np.zeros_like(wall_x)
    
    fig.add_trace(go.Scatter3d(
        x=wall_x, y=wall_y, z=wall_z,
        mode='lines',
        line=dict(color='#2d5016', width=8),
        hoverinfo='skip',
        showlegend=False
    ))
    
    bases = np.array([
        [0, 0, 0],
        [82.5, 82.5, 0],
        [0, 165, 0],
        [-82.5, 82.5, 0],
        [0, 0, 0]
    ])
    
    fig.add_trace(go.Scatter3d(
        x=bases[:, 0], y=bases[:, 1], z=bases[:, 2],
        mode='lines+markers',
        line=dict(color='white', width=5),
        marker=dict(size=6, color='white'),
        hoverinfo='skip',
        showlegend=False
    ))
    
    foul_line_distance = 300
    first_base_x, first_base_y = 82.5, 82.5
    third_base_x, third_base_y = -82.5, 82.5
    
    #foul lines
    fig.add_trace(go.Scatter3d(
        x=[first_base_x, foul_line_distance], 
        y=[first_base_y, foul_line_distance], 
        z=[0, 0],
        mode='lines',
        line=dict(color='#FFD700', width=17, dash='dash'),
        hoverinfo='skip',
        showlegend=False,
        name='Right Field Line'
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[third_base_x, -foul_line_distance], 
        y=[third_base_y, foul_line_distance], 
        z=[0, 0],
        mode='lines',
        line=dict(color='#FFD700', width=17, dash='dash'),
        hoverinfo='skip',
        showlegend=False,
        name='Left Field Line'
    ))
    
    infield_radius = 200
    infield_theta = np.linspace(np.radians(45), np.radians(135), 30)
    infield_x = infield_radius * np.cos(infield_theta)
    infield_y = infield_radius * np.sin(infield_theta)
    
    fig.add_trace(go.Scatter3d(
        x=infield_x, y=infield_y, z=np.zeros_like(infield_x),
        mode='lines',
        line=dict(color='#8B7355', width=3),
        hoverinfo='skip',
        showlegend=False
    ))
    
    fig.update_layout(
        height=800,
        paper_bgcolor='#1e1e1e',
        plot_bgcolor='#1e1e1e',
        scene=dict(
            xaxis=dict(
                range=[-380, 380],
                showbackground=False,
                showticklabels=False,
                showgrid=False,
                showline=False,
                ticks="",
                title=""
            ),
            yaxis=dict(
                range=[-25, 511],
                showbackground=False,
                showticklabels=False,
                showgrid=False,
                showline=False,
                ticks="",
                title=""
            ),
            zaxis=dict(
                range=[-5, 350],
                showbackground=False,
                showticklabels=False,
                showgrid=False,
                showline=False,
                ticks="",
                title=""
            ),
            aspectmode='manual',
            aspectratio=dict(x=2.0, y=1.4, z=0.5),
            camera=dict(
                eye=dict(x=0, y=-1.8, z=1.15)
            )
        ),
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    return fig

def create_strike_zone():
    fig = go.Figure()
    
    #mlb zone dimensions
    zone_width = 1.417
    zone_height = 2.0
    zone_bottom = 1.5
    
    fig.add_shape(
        type="rect",
        x0=-zone_width/2, x1=zone_width/2,
        y0=zone_bottom, y1=zone_bottom + zone_height,
        line=dict(color="white", width=3),
        fillcolor="rgba(255, 255, 255, 0.05)"
    )
    
    for i in range(1, 3):
        x_pos = -zone_width/2 + (zone_width/3) * i
        fig.add_shape(
            type="line",
            x0=x_pos, x1=x_pos,
            y0=zone_bottom, y1=zone_bottom + zone_height,
            line=dict(color="rgba(255,255,255,0.3)", width=1)
        )
        y_pos = zone_bottom + (zone_height/3) * i
        fig.add_shape(
            type="line",
            x0=-zone_width/2, x1=zone_width/2,
            y0=y_pos, y1=y_pos,
            line=dict(color="rgba(255,255,255,0.3)", width=1)
        )
    
    fig.update_layout(
        height=800,
        paper_bgcolor='#1e1e1e',
        plot_bgcolor='#1e1e1e',
        xaxis=dict(
            range=[-1.2, 1.2],
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            title=""
        ),
        yaxis=dict(
            range=[0.8, 4.2],
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            title="",
            scaleanchor="x",
            scaleratio=1
        ),
        showlegend=True,
        legend=dict(
            x=1.02,
            y=1,
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='white',
            borderwidth=1
        ),
        margin=dict(l=20, r=120, t=20, b=20)
    )
    
    return fig

def add_pitch_to_zone(fig, plate_x, plate_z, pitch_type, speed, description, color):
    """Add a single pitch to the strike zone"""
    PITCH_COLORS = {
        'FF': '#FF4444',
        'SI': '#FF6B6B',
        'FC': '#FF8E8E',
        'SL': '#FFA500',
        'CU': '#4169E1',
        'KC': '#6495ED',
        'CH': '#32CD32',
        'FS': '#90EE90',
        'KN': '#9370DB',
    }
    
    PITCH_NAMES = {
        'FF': 'Four-Seam Fastball',
        'SI': 'Sinker',
        'FC': 'Cutter',
        'SL': 'Slider',
        'CU': 'Curveball',
        'KC': 'Knuckle Curve',
        'CH': 'Changeup',
        'FS': 'Splitter',
        'KN': 'Knuckleball',
        'FT': 'Two-Seam Fastball',
        'ST': 'Sweeper',
        'SV': 'Slurve',
        'UN': 'Unknown'
    }
    
    pitch_color = PITCH_COLORS.get(pitch_type, '#CCCCCC')
    pitch_name = PITCH_NAMES.get(pitch_type, pitch_type)
    formatted_description = description.replace('_', ' ').title() if description else 'Unknown'
    hover_text = f"Type: {pitch_name}<br>Speed: {speed:.1f} mph<br>Result: {formatted_description}"
    
    fig.add_trace(go.Scatter(
        x=[plate_x],
        y=[plate_z],
        mode='markers',
        marker=dict(
            size=36,
            color=pitch_color,
            line=dict(color='white', width=1),
            opacity=0.7
        ),
        hovertext=hover_text,
        hoverinfo='text',
        name=pitch_type,
        showlegend=True,
        legendgroup=pitch_type
    ))
    
    return fig

def add_hit_to_field(fig, hc_x, hc_y, launch_angle, launch_speed, distance, description, color='red', hit_type='HR'):
    """Add a single hit trajectory to the field with physics-based landing height"""
    p1 = np.array([0, 0, 1])
    
    #physics for expected distance
    if launch_speed > 0 and launch_angle > 0:
        v_fps = launch_speed * 1.467
        expected_distance = (v_fps ** 2 * np.sin(2 * np.radians(launch_angle))) / 32.2
    else:
        expected_distance = distance
    
    actual_ground_distance = np.sqrt(hc_x**2 + hc_y**2)
    has_rolling = False
    expected_x, expected_y = hc_x, hc_y
    
    # check for rolling on ground balls
    if hit_type not in ['HR', 'home_run', 'hr'] and expected_distance > 0:
        error_pct = abs(actual_ground_distance - expected_distance) / expected_distance
        
        if actual_ground_distance > expected_distance and error_pct > 0.10:
            has_rolling = True
            direction_angle = np.arctan2(hc_y, hc_x) if actual_ground_distance > 0 else 0
            expected_x = expected_distance * np.cos(direction_angle)
            expected_y = expected_distance * np.sin(direction_angle)
    
    landing_height = 0
    
    if hit_type == 'HR' or hit_type == 'home_run':
        if expected_distance > distance * 1.15:
            landing_height = 8 + (expected_distance - distance) * 0.08 * 2.5
        elif expected_distance > distance * 1.05:
            landing_height = 8 + (expected_distance - distance) * 0.05 * 2.5
        else:
            landing_height = 8
    
    arc_end_x = expected_x if has_rolling else hc_x
    arc_end_y = expected_y if has_rolling else hc_y
    p2 = np.array([arc_end_x, arc_end_y, landing_height])
    
    height_scaling_factor = 1.2
    h = height_scaling_factor * np.tan(np.radians(launch_angle)) * np.linalg.norm(p2[:2] - p1[:2])
    h = max(h, 30)
    
    apex = np.array([0.5 * (p1[0] + p2[0]), 0.5 * (p1[1] + p2[1]), h + landing_height])
    
    x, y, z = generate_arc_points(p1, p2, apex, num_points=50)
    
    hover_text = f"<b>{hit_type}</b><br>" + \
                 f"Distance: {distance:.1f} ft<br>" + \
                 f"Launch Angle: {launch_angle:.1f}°<br>" + \
                 f"Exit Velocity: {launch_speed:.1f} mph<br>" + \
                 f"{description}"
    
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines',
        line=dict(color=color, width=4),
        hoverinfo='text',
        hovertext=[hover_text] * len(x),
        showlegend=False
    ))
    
    if has_rolling:
        fig.add_trace(go.Scatter3d(
            x=[arc_end_x, hc_x],
            y=[arc_end_y, hc_y],
            z=[0, 0],
            mode='lines',
            line=dict(color=color, width=4),
            hoverinfo='text',
            hovertext=[hover_text, hover_text],
            showlegend=False
        ))
    
    final_x = hc_x
    final_y = hc_y
    final_z = landing_height if not has_rolling else 0
    
    fig.add_trace(go.Scatter3d(
        x=[final_x], y=[final_y], z=[final_z],
        mode='markers',
        marker=dict(size=8, color=color, symbol='circle'),
        hoverinfo='text',
        hovertext=hover_text,
        showlegend=False
    ))
    
    return fig

def create_strike_zone_plot():
    """Create 2D strike zone visualization for pitchers"""
    fig = go.Figure()
    
    # Strike zone dimensions (approximation in inches)
    zone_width = 17
    zone_height_bottom = 1.5
    zone_height_top = 3.5
    
    fig.add_shape(
        type="rect",
        x0=-zone_width/2/12, y0=zone_height_bottom,
        x1=zone_width/2/12, y1=zone_height_top,
        line=dict(color="white", width=3),
    )
    
    # Add zone divisions (9 zones)
    for i in range(1, 3):
        # Vertical lines
        x = -zone_width/2/12 + i * (zone_width/12) / 3
        fig.add_shape(
            type="line",
            x0=x, y0=zone_height_bottom,
            x1=x, y1=zone_height_top,
            line=dict(color="gray", width=1, dash="dash"),
        )
        # /horizontal lines
        y = zone_height_bottom + i * (zone_height_top - zone_height_bottom) / 3
        fig.add_shape(
            type="line",
            x0=-zone_width/2/12, y0=y,
            x1=zone_width/2/12, y1=y,
            line=dict(color="gray", width=1, dash="dash"),
        )
    
    fig.update_layout(
        height=600,
        paper_bgcolor='#1e1e1e',
        plot_bgcolor='#2d2d2d',
        xaxis=dict(
            range=[-3, 3],
            title="Horizontal Position (ft)",
            showgrid=True,
            gridcolor='#3d3d3d'
        ),
        yaxis=dict(
            range=[0, 5],
            title="Height (ft)",
            showgrid=True,
            gridcolor='#3d3d3d'
        ),
        showlegend=True,
        margin=dict(l=50, r=20, t=40, b=50)
    )
    
    return fig

try:
    hitters_df = pd.read_csv('hitters_data.csv')
    display_cols = ['player_name', 'Team', 'Pos', 'ba', 'obp', 'slg', 'OPS', 'woba',
                   'hrs', 'RBI', 'hits', 'doubles', 'triples', 'abs', 'BB', 'player_id']
    
    available_cols = [col for col in display_cols if col in hitters_df.columns]
    hitters_df = hitters_df[available_cols]
    
    hitters_df = hitters_df.rename(columns={
        'player_name': 'Name',
        'ba': 'AVG',
        'slg': 'SLG',
        'woba': 'wOBA',
        'hrs': 'HR',
        'doubles': '2B',
        'triples': '3B',
        'hits': 'H',
        'abs': 'AB',
        'obp': 'OBP',
        'player_id': 'ID'
    })
    
except FileNotFoundError:
    hitters_df = pd.DataFrame()
    print("hitters_data.csv not found")

# Load team ID mapping for Fangraphs API
team_id_map = {}
try:
    team_df = pd.read_csv('team_idfg.csv')
    team_id_map = dict(zip(team_df['team'], team_df['teamIDfg'].astype(str)))
    print(f"Loaded team ID mapping: {len(team_id_map)} teams")
except Exception as e:
    print(f"Failed to load team_idfg.csv: {e}")

# Mapping from  team abbreviations to pybaseball's team abbreviations
TEAM_ABBR_MAP = {
    'CWS': 'CHW',
    'TB': 'TBR',
    'KC': 'KCR',
    'SD': 'SDP',
    'SF': 'SFG',
    'WSH': 'WSN',
    #  other teams use the same abbreviation
}

STAT_COLS = ['AVG', 'OPS', 'SLG', 'OBP', 'wOBA', 'HR']
MONTHLY_STAT_COLS = ['AVG', 'SLG', 'OBP', 'wOBA', 'HR']  # for bi-monthly splits (no OPS needed)

#  unicode characters (handles accents)
def normalize_name(name):
    if pd.isna(name):
        return ""
    return unicodedata.normalize('NFD', str(name)).encode('ascii', 'ignore').decode('utf-8')

import threading
cache_queue = set()
cache_lock = threading.Lock()

def background_cache_worker():
    while True:
        player_id = None
        with cache_lock:
            if cache_queue:
                player_id = cache_queue.pop()
        
        if player_id and PYBASEBALL_AVAILABLE:
            try:
                # Check if already cached
                cache_file = Path('cache') / f'player_{int(player_id)}_2025_hits.pkl'
                if not cache_file.exists():
                    get_player_hit_data(int(player_id), year=2025, use_cache=False)
            except Exception as e:
                print(f"Background caching failed for player {player_id}: {e}")
        
        import time
        time.sleep(0.5)  # Small delay between cache operations

# background cache worker
if PYBASEBALL_AVAILABLE:
    cache_thread = threading.Thread(target=background_cache_worker, daemon=True)
    cache_thread.start()

try:
    pitchers_df = pd.read_csv('pitchers_data.csv')
    # rename columns for display
    pitcher_display_cols = ['player_name', 'team_name', 'era', 'whip', 'strikeouts', 'walks',
                           'innings_pitched', 'wins', 'losses', 'saves', 'games_played', 'games_started', 'player_id']
    
    available_pitcher_cols = [col for col in pitcher_display_cols if col in pitchers_df.columns]
    pitchers_display = pitchers_df[available_pitcher_cols].copy()
    
    # Calculate K/9 and BB/9 myself i guess smh
    if 'strikeouts' in pitchers_display.columns and 'innings_pitched' in pitchers_display.columns:
        pitchers_display['K/9'] = (pitchers_display['strikeouts'] / pitchers_display['innings_pitched'] * 9).round(2)
    if 'walks' in pitchers_display.columns and 'innings_pitched' in pitchers_display.columns:
        pitchers_display['BB/9'] = (pitchers_display['walks'] / pitchers_display['innings_pitched'] * 9).round(2)
    
    pitchers_display = pitchers_display.rename(columns={
        'player_name': 'Name',
        'team_name': 'Team',
        'era': 'ERA',
        'whip': 'WHIP',
        'strikeouts': 'K',
        'walks': 'BB',
        'innings_pitched': 'IP',
        'wins': 'W',
        'losses': 'L',
        'saves': 'SV',
        'games_played': 'G',
        'games_started': 'GS',
        'player_id': 'ID'
    })
    
    pitchers_df = pitchers_display
except FileNotFoundError:
    pitchers_df = pd.DataFrame()
    print("pitchers_data.csv not found - using hitters only for now")

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("2025 Baseball Statistics Visualizer", 
                   className="text-center mb-4 mt-3",
                   style={'color': '#ffffff'})
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Player Statistics", className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.RadioItems(
                                id='player-type-radio',
                                options=[
                                    {'label': 'Hitters', 'value': 'hitters'},
                                    {'label': 'Pitchers', 'value': 'pitchers'}
                                ],
                                value='hitters',
                                inline=True,
                                className="mb-2"
                            )
                        ], width=12)
                    ]),
                    dbc.Input(
                        id='player-search',
                        placeholder="Search player name...",
                        type="text",
                        className="mb-2"
                    )
                ]),
                dbc.CardBody([
                    dash_table.DataTable(
                        id='player-table',
                        style_table={'overflowY': 'auto', 'maxHeight': '70vh'},
                        style_cell={
                            'textAlign': 'left',
                            'backgroundColor': '#2d2d2d',
                            'color': 'white',
                            'border': '1px solid #444',
                            'minWidth': '60px',
                            'maxWidth': '180px',
                            'overflow': 'hidden',
                            'textOverflow': 'ellipsis'
                        },
                        style_cell_conditional=[
                            {'if': {'column_id': 'Name'}, 'minWidth': '140px', 'maxWidth': '180px'},
                            {'if': {'column_id': 'Team'}, 'minWidth': '65px', 'maxWidth': '70px'},
                            {'if': {'column_id': 'Pos'}, 'minWidth': '50px', 'maxWidth': '55px'},
                            {'if': {'column_id': ['AVG', 'OBP', 'SLG', 'OPS', 'wOBA']}, 'minWidth': '60px', 'maxWidth': '70px'},
                            {'if': {'column_id': ['HR', '2B', '3B', 'H', 'AB', 'BB', 'RBI']}, 'minWidth': '50px', 'maxWidth': '55px'}
                        ],
                        style_header={
                            'backgroundColor': '#1e1e1e',
                            'fontWeight': 'bold',
                            'border': '1px solid #444',
                            'cursor': 'pointer',
                            'overflow': 'hidden',
                            'textOverflow': 'ellipsis'
                        },
                        style_data_conditional=[
                            {
                                'if': {'state': 'selected'},
                                'backgroundColor': '#0d6efd',
                                'border': '1px solid #0d6efd'
                            }
                        ],
                        row_selectable='single',
                        selected_rows=[],
                        page_action='native',
                        page_current=0,
                        page_size=20,
                        sort_action='custom',
                        sort_mode='single',
                        sort_by=[]
                    ),
                    html.Hr(className="my-3"),
                    html.Div([
                        html.H6("About This Visualization", className="mb-2", style={'color': '#00bc8c'}),
                        html.P([
                            "This tool tries to bridge the gap between box scores and actual performance by visualizing MLB Statcast data. ",
                            "Statcast is a tracking system that captures the physics of every pitch and batted ball in the 2025 season. ",
                            "You can see where hitters place the ball across the field and where pitchers locate their pitches in the strike zone. ",
                            "Some hitters show balanced distributions while others have pronounced pull tendencies. ",
                            "Observe home run trajectories to reveal power variations-- some barely clear the wall with line drives while others launch high arcs deep into the seats. ",
                            "The comparative analysis shows how players perform relative to their team and league averages, ",
                            "and bi-monthly splits reveal performance changes throughout the season."
                        ], className="mb-2", style={'fontSize': '0.85rem'}),
                        html.P([
                            html.Strong("How to Use: "),
                            "Toggle between Hitters and Pitchers, search for players by name, then select a player ",
                            "using the radio button to the left of their name. ",
                            "For hitters, use the checkboxes to filter hit types and hover over trajectories for details. ",
                            "For pitchers, hover over pitch locations to see pitch type and velocity. ",
                            "Try rotating the 3D field and exploring the statistics below to discover interesting patterns."
                        ], style={'fontSize': '0.85rem', 'marginBottom': '0'})
                    ], style={'padding': '0.5rem'})
                ])
            ], className="h-100")
        ], width=5),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    dbc.Row([
                        dbc.Col([
                            html.H4("Visualization", className="mb-0")
                        ], width=6),
                        dbc.Col([
                            dbc.Checklist(
                                id='hit-type-checklist',
                                options=[
                                    {'label': ' HR', 'value': 'HR'},
                                    {'label': ' 3B', 'value': '3B'},
                                    {'label': ' 2B', 'value': '2B'},
                                    {'label': ' 1B', 'value': '1B'}
                                ],
                                value=['HR'],
                                inline=True,
                                switch=True,
                                style={'display': 'block'}
                            ),
                            dcc.Dropdown(
                                id='pitch-range-selector',
                                options=[],
                                value=0,
                                placeholder='Select pitch range',
                                style={'display': 'none', 'marginTop': '5px'}
                            )
                        ], width=6)
                    ])
                ]),
                dbc.CardBody([
                    dcc.Loading(
                        id="loading-viz",
                        type="default",
                        children=dcc.Graph(
                            id='main-visualization',
                            figure=create_3d_baseball_field(),
                            config={'displayModeBar': True, 'displaylogo': False},
                            style={'height': '800px'}
                        )
                    )
                ], style={'padding': '0.5rem'})
            ], className="mb-3"),
            
            dbc.Card([
                dbc.CardHeader([
                    dbc.Row([
                        dbc.Col([
                            html.H4(id='comparison-title', children="Comparative Analysis", className="mb-0")
                        ], width=6),
                        dbc.Col([
                            html.H5("Bi-Monthly Splits", className="mb-0 text-center", style={'fontSize': '1.15rem'})
                        ], width=6)
                    ])
                ]),
                dbc.CardBody([
                    html.Div(id='comparison-loading', children=[
                        html.P("Select a player to view comparative statistics", 
                              className="text-center text-muted mt-4")
                    ], style={'display': 'block'}),
                    dcc.Loading(
                        id="loading-comparison",
                        type="circle",
                        color="#00bc8c",
                        children=[
                            html.Div([
                                dbc.Row([
                                    dbc.Col([
                                        html.Div(id='comparison-vs-team'),
                                        html.Div(id='comparison-vs-league', style={'display': 'none'})
                                    ], width=6),
                                    dbc.Col([
                                        html.Div(id='bimonthly-splits')
                                    ], width=6)
                                ])
                            ], id='comparison-content', style={'display': 'none'})
                        ]
                    )
                ])
            ])
        ], width=7)
    ], className="mb-4")
], fluid=True, style={'backgroundColor': '#1a1a1a', 'minHeight': '100vh', 'padding': '20px'})

def get_league_averages():
    # Calculate league averages from all teams
    try:
        all_teams = team_batting(2025, stat_columns=STAT_COLS)
        return {
            'AVG': all_teams['AVG'].mean(),
            'OPS': all_teams['OPS'].mean(),
            'SLG': all_teams['SLG'].mean(),
            'OBP': all_teams['OBP'].mean(),
            'wOBA': all_teams['wOBA'].mean(),
            'HR': all_teams['HR'].sum()
        }
    except Exception as e:
        print(f"Error fetching league averages: {e}")
        return None

def get_team_averages(team_abbr):
    try:
        team_id = team_id_map.get(team_abbr)
        if not team_id:
            print(f"Team {team_abbr} not found in mapping")
            return None
        
        pybaseball_abbr = TEAM_ABBR_MAP.get(team_abbr, team_abbr)
        
        all_teams = team_batting(2025, stat_columns=STAT_COLS)
        team_data = all_teams[all_teams['Team'] == pybaseball_abbr]
        
        if team_data.empty:
            print(f"Team {pybaseball_abbr} (from {team_abbr}) not found in pybaseball data")
            return None
        
        return {
            'AVG': team_data['AVG'].values[0],
            'OPS': team_data['OPS'].values[0],
            'SLG': team_data['SLG'].values[0],
            'OBP': team_data['OBP'].values[0],
            'wOBA': team_data['wOBA'].values[0],
            'HR': team_data['HR'].values[0]
        }
    except Exception as e:
        print(f"Error fetching team averages for {team_abbr}: {e}")
        return None

def get_league_pitcher_averages():
    try:
        all_teams = team_pitching(2025)
        return {
            'ERA': all_teams['ERA'].mean(),
            'WHIP': all_teams['WHIP'].mean(),
            'K/9': (all_teams['SO'].sum() / all_teams['IP'].sum() * 9),
            'BB/9': (all_teams['BB'].sum() / all_teams['IP'].sum() * 9)
        }
    except Exception as e:
        print(f"Error fetching league pitcher averages: {e}")
        return None

def get_team_pitcher_averages(team_abbr):
    try:
        team_id = team_id_map.get(team_abbr)
        if not team_id:
            print(f"Team {team_abbr} not found in mapping")
            return None
        
        pybaseball_abbr = TEAM_ABBR_MAP.get(team_abbr, team_abbr)
        
        all_teams = team_pitching(2025)
        team_data = all_teams[all_teams['Team'] == pybaseball_abbr]
        
        if team_data.empty:
            print(f"Team {pybaseball_abbr} (from {team_abbr}) not found in pybaseball pitching data")
            return None
        
        ip = team_data['IP'].values[0]
        k9 = (team_data['SO'].values[0] / ip * 9) if ip > 0 else 0
        bb9 = (team_data['BB'].values[0] / ip * 9) if ip > 0 else 0
        
        return {
            'ERA': team_data['ERA'].values[0],
            'WHIP': team_data['WHIP'].values[0],
            'K/9': k9,
            'BB/9': bb9
        }
    except Exception as e:
        print(f"Error fetching team pitcher averages for {team_abbr}: {e}")
        return None

def get_player_monthly_stats(player_name, team_abbr, position, month):
    try:
        team_id = team_id_map.get(team_abbr, '')
        
        # team filter first
        if team_id:
            try:
                data = batting_stats(2025, month=month, team=team_id, stat_columns=MONTHLY_STAT_COLS, qual=1)
                if isinstance(data, pd.DataFrame) and 'Name' in data.columns and len(data) > 0:
                    data['Name_normalized'] = data['Name'].apply(normalize_name)
                    player_name_normalized = normalize_name(player_name)
                    
                    player_data = data[data['Name'] == player_name]
                    if player_data.empty:
                        player_data = data[data['Name_normalized'] == player_name_normalized]
                    if not player_data.empty:
                        has_stats = all(col in player_data.columns for col in ['AVG', 'SLG', 'OBP', 'wOBA', 'HR'])
                        if has_stats:
                            return {
                                'AVG': player_data['AVG'].values[0] if 'AVG' in player_data else None,
                                'SLG': player_data['SLG'].values[0] if 'SLG' in player_data else None,
                                'OBP': player_data['OBP'].values[0] if 'OBP' in player_data else None,
                                'wOBA': player_data['wOBA'].values[0] if 'wOBA' in player_data else None,
                                'HR': player_data['HR'].values[0] if 'HR' in player_data else 0
                            }
            except Exception as e:
                print(f"Team filter failed for {player_name} (team {team_id}, month {month}): {e}")
        
        #fallback to position filter
        try:
            data = batting_stats(2025, month=month, position=position, stat_columns=MONTHLY_STAT_COLS, qual=1)
            if isinstance(data, pd.DataFrame) and 'Name' in data.columns and len(data) > 0:
                data['Name_normalized'] = data['Name'].apply(normalize_name)
                player_name_normalized = normalize_name(player_name)
                
                player_data = data[data['Name'] == player_name]
                if player_data.empty:
                    player_data = data[data['Name_normalized'] == player_name_normalized]
                if not player_data.empty:
                    has_stats = all(col in player_data.columns for col in ['AVG', 'SLG', 'OBP', 'wOBA', 'HR'])
                    if has_stats:
                        return {
                            'AVG': player_data['AVG'].values[0] if 'AVG' in player_data else None,
                            'SLG': player_data['SLG'].values[0] if 'SLG' in player_data else None,
                            'OBP': player_data['OBP'].values[0] if 'OBP' in player_data else None,
                            'wOBA': player_data['wOBA'].values[0] if 'wOBA' in player_data else None,
                            'HR': player_data['HR'].values[0] if 'HR' in player_data else 0
                        }
        except Exception as e:
            print(f"Position filter failed for {player_name} (pos {position}, month {month}): {e}")
        
        return None
        
    except Exception as e:
        print(f"Error fetching monthly stats for {player_name}, month {month}: {e}")
        return None

def format_stat_with_diff(value, comparison, stat_name, show_diff_on_comparison=True):
    if value is None or comparison is None or pd.isna(value) or pd.isna(comparison):
        return html.Span("--")
    
    diff = value - comparison
    
    if stat_name == 'HR':
        value_str = f"{int(value)}"
        comp_str = f"{int(comparison)}"
        diff_str = f"{int(diff):+d}"
    else:
        value_str = f"{value:.3f}"
        comp_str = f"{comparison:.3f}"
        diff_str = f"{diff:+.3f}"
    
    color = '#28a745' if diff > 0 else '#dc3545' if diff < 0 else '#6c757d'
    
    if show_diff_on_comparison:
        return html.Div([
            html.Span(comp_str, style={'marginRight': '5px'}),
            html.Span(diff_str, style={'color': color, 'fontSize': '0.85em'})
        ])
    else:
        return html.Span(value_str, style={'fontWeight': 'bold'})

def create_comparison_table(player_stats, team_stats, league_stats):
    stats_to_show = ['AVG', 'OPS', 'SLG', 'OBP', 'wOBA']
    
    rows = []
    for stat in stats_to_show:
        player_val = player_stats.get(stat)
        team_val = team_stats.get(stat)
        league_val = league_stats.get(stat)
        
        rows.append(
            html.Tr([
                html.Td(stat, style={'fontWeight': 'bold', 'width': '60px', 'paddingRight': '10px'}),
                html.Td(format_stat_with_diff(player_val, player_val, stat, show_diff_on_comparison=False), 
                       style={'width': '80px', 'paddingRight': '10px'}),
                html.Td(format_stat_with_diff(player_val, team_val, stat, show_diff_on_comparison=True),
                       style={'width': '110px', 'paddingRight': '10px'}),
                html.Td(format_stat_with_diff(player_val, league_val, stat, show_diff_on_comparison=True),
                       style={'width': '110px'})
            ])
        )
    
    header = html.Tr([
        html.Th('', style={'width': '60px'}),
        html.Th('Player', style={'width': '80px', 'fontWeight': 'bold', 'fontSize': '0.9em'}),
        html.Th('Team Avg', style={'width': '110px', 'fontWeight': 'bold', 'fontSize': '0.9em'}),
        html.Th('League Avg', style={'width': '110px', 'fontWeight': 'bold', 'fontSize': '0.9em'})
    ])
    
    return html.Table(
        [html.Thead(header), html.Tbody(rows)], 
        style={'width': '100%', 'fontSize': '0.95em'}
    )

def create_bimonthly_splits_table(periods_data):
    """Create condensed table for bi-monthly splits"""
    stats_to_show = ['AVG', 'SLG', 'OBP', 'wOBA', 'HR']
    period_names = ['Apr + May', 'Jun + Jul', 'Aug + Sep']
    
    header = html.Tr([
        html.Th('', style={'width': '70px', 'padding': '10px', 'borderBottom': '2px solid #444'})
    ] + [
        html.Th(period, style={'fontWeight': 'bold', 'fontSize': '0.9em', 'padding': '10px', 'textAlign': 'center', 'borderBottom': '2px solid #444'})
        for period in period_names
    ])
    
    rows = []
    for idx, stat in enumerate(stats_to_show):
        bg_color = '#2d2d2d' if idx % 2 == 0 else '#333333'
        cells = [html.Td(stat, style={'fontWeight': 'bold', 'padding': '10px', 'backgroundColor': bg_color})]
        for period in period_names:
            val = periods_data.get(period, {}).get(stat)
            if val is not None and not pd.isna(val):
                val_str = f"{int(val)}" if stat == 'HR' else f"{val:.3f}"
            else:
                val_str = "--"
            cells.append(html.Td(val_str, style={'padding': '10px', 'textAlign': 'center', 'backgroundColor': bg_color}))
        rows.append(html.Tr(cells))
    
    return html.Table(
        [html.Thead(header), html.Tbody(rows)],
        style={'width': '100%', 'fontSize': '0.95em', 'borderCollapse': 'collapse'}
    )

@app.callback(
    [Output('player-table', 'data'),
     Output('player-table', 'columns')],
    [Input('player-type-radio', 'value'),
     Input('player-search', 'value'),
     Input('player-table', 'sort_by')]
)
def update_table(player_type, search_query, sort_by):
    """Update table based on player type and search query"""
    df = hitters_df.copy() if player_type == 'hitters' else pitchers_df.copy()
    
    if df.empty:
        return [], []
    
    if search_query and search_query.strip():
        df = df[df['Name'].str.contains(search_query, case=False, na=False)]
    
    #sorting logic
    if sort_by and len(sort_by) > 0:
        col = sort_by[0]['column_id']
        direction = sort_by[0]['direction']
        ascending = (direction == 'desc')
        df = df.sort_values(by=col, ascending=ascending)
    
    columns = [{"name": col, "id": col} for col in df.columns if col != 'ID']
    
    return df.to_dict('records'), columns

@app.callback(
    Output('player-table', 'page_current', allow_duplicate=True),
    [Input('player-table', 'page_current'),
     Input('player-type-radio', 'value')],
    [State('player-table', 'data')],
    prevent_initial_call=True
)
def cache_page_players(page_current, player_type, table_data):
    """Queue players on current page for background caching"""
    if player_type == 'hitters' and table_data:
        page_size = 20
        start_idx = page_current * page_size
        end_idx = start_idx + page_size
        
        with cache_lock:
            # Queue players on current page
            for i in range(start_idx, min(end_idx, len(table_data))):
                if i < len(table_data):
                    player = table_data[i]
                    player_id = player.get('ID')
                    if player_id and pd.notna(player_id):
                        cache_queue.add(int(player_id))
    
    return page_current

@app.callback(
    [Output('pitch-range-selector', 'options'),
     Output('pitch-range-selector', 'value'),
     Output('pitch-range-selector', 'style')],
    [Input('player-table', 'selected_rows'),
     Input('player-type-radio', 'value')],
    [State('player-table', 'data'),
     State('pitch-range-selector', 'value')]
)
def update_pitch_selector(selected_rows, player_type, table_data, current_value):
    # Hide by default
    if player_type != 'pitchers' or not selected_rows or not table_data:
        return [], 0, {'display': 'none', 'marginTop': '5px'}
    
    selected_player = table_data[selected_rows[0]]
    player_id = selected_player.get('ID')
    
    if not player_id:
        return [], 0, {'display': 'none', 'marginTop': '5px'}
    
    # Load pitch data to check count
    pitch_data = get_player_pitch_data(int(player_id), year=2025)
    
    if pitch_data is None or pitch_data.empty:
        return [], 0, {'display': 'none', 'marginTop': '5px'}
    
    total_pitches = len(pitch_data)
    
    # Only show selector if more than 200 pitches
    if total_pitches <= 200:
        return [], 0, {'display': 'none', 'marginTop': '5px'}
    
    # Create options for pitch ranges (200 at a time)
    options = []
    for i in range(0, total_pitches, 200):
        end = min(i + 200, total_pitches)
        label = f'Pitches {i+1}-{end}'
        options.append({'label': label, 'value': i})
    
    # Show the selector and default to first range
    return options, 0, {'display': 'block', 'marginTop': '5px'}

@app.callback(
    Output('main-visualization', 'figure'),
    [Input('player-table', 'selected_rows'),
     Input('player-type-radio', 'value'),
     Input('hit-type-checklist', 'value'),
     Input('pitch-range-selector', 'value')],
    [State('player-table', 'data')]
)
def update_visualization(selected_rows, player_type, hit_types, pitch_range_start, table_data):
    """Update the main visualization based on selected player"""
    if not selected_rows or not table_data:
        # Return empty field or strike zone
        if player_type == 'hitters':
            return create_3d_baseball_field()
        else:
            return create_strike_zone()
    
    selected_player = table_data[selected_rows[0]]
    player_name = selected_player.get('Name', 'Unknown')
    
    if player_type == 'hitters':
        # 3D field
        fig = create_3d_baseball_field()
        
        player_id = selected_player.get('ID')
        hit_data = None
        
        if player_id:
            hit_data = get_player_hit_data(int(player_id), year=2025)
        
        if hit_data is not None and not hit_data.empty:
            hit_colors = {
                'home_run': 'red',
                'triple': 'orange',
                'double': 'yellow',
                'single': 'lightblue'
            }
            
            hit_type_map = {
                'HR': 'home_run',
                '3B': 'triple',
                '2B': 'double',
                '1B': 'single'
            }
        
            events_to_show = [hit_type_map[ht] for ht in hit_types if ht in hit_type_map]
            
            for event_type in events_to_show:
                event_hits = hit_data[hit_data['events'] == event_type].copy()
                
                for _, hit in event_hits.iterrows():
                    # Lame math stuff from the internet
                    if pd.notna(hit.get('hc_x')) and pd.notna(hit.get('hc_y')):
                        HOME_X = 125.42
                        HOME_Y = 198.27
                        FEET_PER_UNIT = 2.5
                        
                        hc_x = (hit['hc_x'] - HOME_X) * FEET_PER_UNIT
                        hc_y = (HOME_Y - hit['hc_y']) * FEET_PER_UNIT
                        
                        launch_angle = hit.get('launch_angle', 25)
                        launch_speed = hit.get('launch_speed', 95)
                        hit_distance = hit.get('hit_distance_sc', 300)
                        description = hit.get('des', f"{event_type} by {player_name}")
                        
                        pitcher_id = hit.get('pitcher')
                        pitcher_name = 'Unknown'
                        if pitcher_id and pd.notna(pitcher_id) and not pitchers_df.empty:
                            if 'ID' in pitchers_df.columns:
                                pitcher_match = pitchers_df[pitchers_df['ID'] == int(pitcher_id)]
                                if not pitcher_match.empty:
                                    pitcher_name = pitcher_match.iloc[0]['Name']
                        
                        # hover my beloved
                        hover_desc = f"{description}<br>vs {pitcher_name}"
                        
                        fig = add_hit_to_field(
                            fig, hc_x, hc_y, launch_angle, launch_speed, 
                            hit_distance, hover_desc,
                            color=hit_colors[event_type],
                            hit_type=event_type.upper().replace('HOME_RUN', 'HR').replace('_', '')
                        )
        return fig
    else:
        fig = create_strike_zone()
        
        player_id = selected_player.get('ID')
        pitch_data = None
        
        if player_id:
            pitch_data = get_player_pitch_data(int(player_id), year=2025)
        
        if pitch_data is not None and not pitch_data.empty:
            total_pitches = len(pitch_data)
            start_idx = pitch_range_start if pitch_range_start is not None else 0
            end_idx = min(start_idx + 200, total_pitches)
            pitches_to_show = pitch_data.iloc[start_idx:end_idx]
            
            seen_types = set()
            
            for _, pitch in pitches_to_show.iterrows():
                if pd.notna(pitch.get('plate_x')) and pd.notna(pitch.get('plate_z')):
                    plate_x = pitch['plate_x']
                    plate_z = pitch['plate_z']
                    pitch_type = pitch.get('pitch_type', 'UN')
                    speed = pitch.get('release_speed', 0)
                    description = pitch.get('description', 'Unknown')
                    
                    if pitch_type in seen_types:
                        showlegend = False
                    else:
                        seen_types.add(pitch_type)
                        showlegend = True
                    
                    fig = add_pitch_to_zone(fig, plate_x, plate_z, pitch_type, speed, description, None)
                    fig.data[-1].showlegend = showlegend
        
        return fig



@app.callback(
    [Output('comparison-vs-team', 'children'),
     Output('comparison-vs-league', 'children'),
     Output('bimonthly-splits', 'children'),
     Output('comparison-loading', 'style'),
     Output('comparison-content', 'style'),
     Output('comparison-title', 'children')],
    [Input('player-table', 'selected_rows'),
     Input('player-table', 'data'),
     Input('player-type-radio', 'value')]
)
def update_comparative_analysis(selected_rows, table_data, player_type):
    if not selected_rows or not table_data:
        empty_msg = html.P("Select a player to view analysis", className="text-center text-muted")
        return empty_msg, empty_msg, empty_msg, {'display': 'block'}, {'display': 'none'}, "Comparative Analysis"
    
    if selected_rows[0] >= len(table_data):
        empty_msg = html.P("Select a player to view analysis", className="text-center text-muted")
        return empty_msg, empty_msg, empty_msg, {'display': 'block'}, {'display': 'none'}, "Comparative Analysis"
    
    selected_row = table_data[selected_rows[0]]
    player_name = selected_row['Name']
    
    if player_type == 'pitchers':
        team = selected_row.get('Team', '')
        
        pitcher_stats = {
            'ERA': selected_row.get('ERA'),
            'WHIP': selected_row.get('WHIP'),
            'K/9': selected_row.get('K/9'),
            'BB/9': selected_row.get('BB/9')
        }
        
        team_pitcher_stats = get_team_pitcher_averages(team)
        if not team_pitcher_stats:
            team_pitcher_stats = {k: None for k in pitcher_stats.keys()}
        
        league_pitcher_stats = get_league_pitcher_averages()
        if not league_pitcher_stats:
            league_pitcher_stats = {k: None for k in pitcher_stats.keys()}

        stats_to_show = ['ERA', 'WHIP', 'K/9', 'BB/9', 'W-L']
        rows = []
        for stat in stats_to_show:
            if stat == 'W-L':
                w = selected_row.get('W', 0)
                l = selected_row.get('L', 0)
                player_str = f"{w}-{l}"
                rows.append(
                    html.Tr([
                        html.Td(stat, style={'fontWeight': 'bold', 'width': '60px', 'paddingRight': '10px'}),
                        html.Td(player_str, style={'width': '80px', 'paddingRight': '10px', 'fontWeight': 'bold'}),
                        html.Td('', style={'width': '110px', 'paddingRight': '10px'}),
                        html.Td('', style={'width': '110px'})
                    ])
                )
                continue
            
            player_val = pitcher_stats.get(stat)
            team_val = team_pitcher_stats.get(stat)
            league_val = league_pitcher_stats.get(stat)
            
            is_lower_better = stat in ['ERA', 'WHIP', 'BB/9']
            
            player_str = f"{player_val:.2f}" if player_val is not None and not pd.isna(player_val) else "--"
            
            if player_val is not None and team_val is not None and not pd.isna(player_val) and not pd.isna(team_val):
                diff = player_val - team_val
                if is_lower_better:
                    color = '#28a745' if diff < 0 else '#dc3545' if diff > 0 else '#6c757d'
                else:
                    color = '#28a745' if diff > 0 else '#dc3545' if diff < 0 else '#6c757d'
                team_cell = html.Div([
                    html.Span(f"{team_val:.2f}", style={'marginRight': '5px'}),
                    html.Span(f"{diff:+.2f}", style={'color': color, 'fontSize': '0.85em'})
                ])
            else:
                team_cell = html.Span("--")
            
            if player_val is not None and league_val is not None and not pd.isna(player_val) and not pd.isna(league_val):
                diff = player_val - league_val
                if is_lower_better:
                    color = '#28a745' if diff < 0 else '#dc3545' if diff > 0 else '#6c757d'
                else:
                    color = '#28a745' if diff > 0 else '#dc3545' if diff < 0 else '#6c757d'
                league_cell = html.Div([
                    html.Span(f"{league_val:.2f}", style={'marginRight': '5px'}),
                    html.Span(f"{diff:+.2f}", style={'color': color, 'fontSize': '0.85em'})
                ])
            else:
                league_cell = html.Span("--")
            
            rows.append(
                html.Tr([
                    html.Td(stat, style={'fontWeight': 'bold', 'width': '60px', 'paddingRight': '10px'}),
                    html.Td(player_str, style={'width': '80px', 'paddingRight': '10px', 'fontWeight': 'bold'}),
                    html.Td(team_cell, style={'width': '110px', 'paddingRight': '10px'}),
                    html.Td(league_cell, style={'width': '110px'})
                ])
            )
        
        header = html.Tr([
            html.Th('', style={'width': '60px'}),
            html.Th('Pitcher', style={'width': '80px', 'fontWeight': 'bold', 'fontSize': '0.9em'}),
            html.Th('Team Avg', style={'width': '110px', 'fontWeight': 'bold', 'fontSize': '0.9em'}),
            html.Th('League Avg', style={'width': '110px', 'fontWeight': 'bold', 'fontSize': '0.9em'})
        ])
        
        pitcher_comparison = dbc.Card([
            dbc.CardHeader(html.H6("Season Comparison", className="mb-0")),
            dbc.CardBody([
                html.Table(
                    [html.Thead(header), html.Tbody(rows)], 
                    style={'width': '100%', 'fontSize': '0.95em'}
                )
            ], style={'padding': '0.75rem'})
        ], style={'backgroundColor': '#2d2d2d'})
        
        empty_msg = html.P("Bi-monthly splits not available for pitchers", className="text-center text-muted")
        return pitcher_comparison, html.Div(), empty_msg, {'display': 'none'}, {'display': 'block'}, f"Pitcher Analysis - {player_name}"
    
    if player_type != 'hitters':
        empty_msg = html.P("Select a hitter to view comparisons", className="text-center text-muted")
        return empty_msg, empty_msg, empty_msg, {'display': 'block'}, {'display': 'none'}, "Comparative Analysis"
    
    if not PYBASEBALL_AVAILABLE:
        error_msg = html.P("PyBaseball not available", className="text-center text-muted")
        return error_msg, error_msg, error_msg, {'display': 'block'}, {'display': 'none'}, "Comparative Analysis"
    
    if selected_rows[0] >= len(table_data):
        empty_msg = html.P("Select a hitter to view comparisons", className="text-center text-muted")
        return empty_msg, empty_msg, empty_msg, {'display': 'block'}, {'display': 'none'}, "Comparative Analysis"
    
    selected_row = table_data[selected_rows[0]]
    player_name = selected_row['Name']
    team = selected_row.get('Team', '')
    position = selected_row.get('Pos', '')
    
    player_stats = {
        'AVG': selected_row.get('AVG'),
        'OPS': selected_row.get('OPS'),
        'SLG': selected_row.get('SLG'),
        'OBP': selected_row.get('OBP'),
        'wOBA': selected_row.get('wOBA'),
        'HR': selected_row.get('HR')
    }
    
    team_stats = get_team_averages(team)
    if not team_stats:
        team_stats = {k: None for k in player_stats.keys()}
    
    league_averages = get_league_averages()
    if not league_averages:
        league_averages = {k: None for k in player_stats.keys()}
    
    # this table is soooo not worth it
    comparison_table = dbc.Card([
        dbc.CardHeader(html.H6("Season Comparison", className="mb-0")),
        dbc.CardBody([
            create_comparison_table(player_stats, team_stats, league_averages)
        ], style={'padding': '0.75rem'})
    ], style={'backgroundColor': '#2d2d2d'})
    
    # you know what, bi-monthly splits maybe worth it
    periods = {
        'Apr + May': ['4', '5'],
        'Jun + Jul': ['6', '7'],
        'Aug + Sep': ['8', '9']
    }
    
    periods_data = {}
    for period_name, months in periods.items():
        period_stats_list = []
        for month in months:
            month_stats = get_player_monthly_stats(player_name, team, position, month)
            if month_stats:
                period_stats_list.append(month_stats)
        
        if period_stats_list:
            def safe_avg(stat_name):
                values = [s[stat_name] for s in period_stats_list if s.get(stat_name) is not None]
                return sum(values) / len(values) if values else None
            
            def safe_sum(stat_name):
                values = [s[stat_name] for s in period_stats_list if s.get(stat_name) is not None]
                return sum(values) if values else None
            
            period_stats = {
                'AVG': safe_avg('AVG'),
                'SLG': safe_avg('SLG'),
                'OBP': safe_avg('OBP'),
                'wOBA': safe_avg('wOBA'),
                'HR': safe_sum('HR')
            }
        else:
            period_stats = {k: None for k in ['AVG', 'SLG', 'OBP', 'wOBA', 'HR']}
        
        periods_data[period_name] = period_stats
    
    splits_table = dbc.Card([
        dbc.CardBody([
            create_bimonthly_splits_table(periods_data)
        ], style={'padding': '0.75rem'})
    ], style={'backgroundColor': '#2d2d2d'})
    
    return (comparison_table, html.Div(), splits_table, 
            {'display': 'none'}, {'display': 'block'}, 
            f"Comparative Analysis - {player_name}")

server = app.server

if __name__ == '__main__':
    app.run(debug=True, port=8050)
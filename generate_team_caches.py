import os
import pickle
from pybaseball import team_batting, team_pitching

# Create cache directory if it doesn't exist
cache_dir = 'cache'
os.makedirs(cache_dir, exist_ok=True)

print("Generating team batting cache...")
try:
    # Get all data without filtering columns
    team_batting_data = team_batting(2025)
    with open(os.path.join(cache_dir, 'team_batting_2025.pkl'), 'wb') as f:
        pickle.dump(team_batting_data, f)
    print(f"✓ Team batting cache created ({len(team_batting_data)} teams)")
except Exception as e:
    print(f"✗ Failed to cache team batting: {e}")

print("\nGenerating team pitching cache...")
try:
    # Get all data without filtering columns
    team_pitching_data = team_pitching(2025)
    with open(os.path.join(cache_dir, 'team_pitching_2025.pkl'), 'wb') as f:
        pickle.dump(team_pitching_data, f)
    print(f"✓ Team pitching cache created ({len(team_pitching_data)} teams)")
except Exception as e:
    print(f"✗ Failed to cache team pitching: {e}")

print("\nDone! Team caches generated successfully.")

"""
Simulate frontend formatTime function to verify timestamp parsing.
"""
from datetime import datetime, timezone, timedelta

# Simulate the API response timestamps
api_timestamps = [
    '2026-11-12T14:01:32.393000Z',  # Old conversation
    '2026-11-13T18:05:35.721000Z',  # New conversation
    '2026-11-13T10:00:00.000000Z',  # Example: 10:00 UTC = 15:30 IST
]

# Simulate current time (browser) - let's say it's 2026-11-13 18:30 UTC (2026-11-14 00:00 IST)
# This is a fixed time for testing
now_utc = datetime(2026, 11, 13, 18, 30, 0, tzinfo=timezone.utc)
now_ist = now_utc.astimezone(timezone(timedelta(hours=5, minutes=30)))

print(f"Simulated browser current time (UTC): {now_utc}")
print(f"Simulated browser current time (IST): {now_ist}")
print()

for ts in api_timestamps:
    print(f"API timestamp: {ts}")
    
    # Frontend parsing: new Date(isoString)
    # In JS: new Date('2026-11-12T14:01:32.393000Z') -> parses as UTC
    parsed = datetime.fromisoformat(ts.replace('Z', '+00:00'))
    print(f"  Parsed as UTC: {parsed}")
    
    # Frontend relative time calculation: now - date (both in ms since epoch)
    # In JS: const now = new Date(); const diffMs = now - date;
    # new Date() in JS returns local time but internal value is UTC ms since epoch
    # So we simulate with UTC times
    diff_ms = (now_utc - parsed).total_seconds() * 1000
    diff_mins = int(diff_ms / 60000)
    diff_hours = int(diff_ms / 3600000)
    
    print(f"  Diff (ms): {diff_ms}")
    print(f"  Diff (mins): {diff_mins}")
    print(f"  Diff (hours): {diff_hours}")
    
    # formatTime logic
    if diff_mins < 1:
        result = 'Just now'
    elif diff_mins < 60:
        result = f'{diff_mins}m ago'
    elif diff_hours < 24:
        result = f'{diff_hours}h ago'
    else:
        # Check yesterday (local calendar day)
        yesterday = now_utc - timedelta(days=1)
        if parsed.date() == yesterday.date():
            result = 'Yesterday'
        else:
            # toLocaleDateString('en-GB') uses LOCAL timezone
            # In IST, the date might be different
            result = parsed.astimezone(timezone(timedelta(hours=5, minutes=30))).strftime('%d %b %Y')
    
    print(f"  formatTime result: {result}")
    
    # Also show what the local date would be
    local_date = parsed.astimezone(timezone(timedelta(hours=5, minutes=30)))
    print(f"  Local (IST) date: {local_date.strftime('%d %b %Y %H:%M')}")
    print()
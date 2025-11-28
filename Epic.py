def group_user_events(logs: list[str]) -> dict:
    events = {}  # user -> list of (timestamp, action)

    for log in logs:
        parts = log.split()

        # must have exactly 3 parts: timestamp user action
        if len(parts) != 3:
            continue

        ts, user, action = parts

        # timestamp must be integer
        if not ts.isdigit():
            continue

        ts = int(ts)

        if user not in events:
            events[user] = []

        events[user].append((ts, action))

    # sort each user's events by timestamp
    for user in events:
        events[user].sort(key=lambda x: x[0])
        # convert (ts, action) → action only
        events[user] = [action for (_, action) in events[user]]

    return events

logs = [
    "100 user1 login",
    "110 user2 click",
    "105 user1 click",
    "bad log",
    "50 user2 start"
]

print(group_user_events(logs))

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



def simulate_patients(arrivals: list[tuple[int,int,int]], t: int) -> dict:
    # Sort arrivals by arrival_time (just in case)
    arrivals.sort(key=lambda x: x[0])

    time = 0                     # current time
    i = 0                        # index for arrivals
    n = len(arrivals)
    waiting = []                # waiting patients: (severity, arrival_time, id)
    result = {}                 # patient_id -> start_time

    while i < n or waiting:
        # Add all patients who arrive at this time
        while i < n and arrivals[i][0] <= time:
            arr_time, pid, sev = arrivals[i]
            waiting.append((sev, arr_time, pid))
            i += 1

        if waiting:
            # Choose highest severity, tie → earliest arrival
            waiting.sort(key=lambda x: (-x[0], x[1]))
            sev, arr_time, pid = waiting.pop(0)

            # Start treatment now
            result[pid] = time

            # Doctor busy for t minutes
            time += t
        else:
            # If no one waiting, jump directly to next arrival time
            time = arrivals[i][0]

    return result

arrivals = [(0,1,5), (1,2,2), (2,3,5)]
t = 4

print(simulate_patients(arrivals, t))

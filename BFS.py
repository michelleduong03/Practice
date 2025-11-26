from collections import deque

def shortest_path(n, edges, start, end):
    # TODO: build adjacency list
    graph = {i: [] for i in range(n)}
    for a, b in edges:
        graph[a].append(b)
        graph[b].append(a)

    queue = deque([(start, 0)])  # (node, distance)
    visited = set([start])

    while queue:
        node, dist = queue.popleft()

        if node == end:
            return dist

        # TODO: BFS neighbors
        for nei in graph[node]:
            if nei not in visited:
                visited.add(nei)
                queue.append((nei, dist + 1))

    return -1   # no path


# ------------ TEST ------------
n = 6
edges = [(0,1),(0,2),(1,3),(2,3),(3,4),(4,5)]
print(shortest_path(n, edges, 0, 5))



from collections import deque

def shortest_path_maze(maze, start, end):
    rows = len(maze)
    cols = len(maze[0])

    queue = deque([(start[0], start[1], 0)])  # (r, c, distance)
    visited = set([start])

    while queue:
        r, c, dist = queue.popleft()

        if (r, c) == end:
            return dist

        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = r + dr, c + dc

            # TODO: check bounds and walls
            if 0 <= nr < rows and 0 <= nc < cols:
                if maze[nr][nc] == 0 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc, dist + 1))

    return -1  # no path


# ------------ TEST ------------
maze = [
    [0,0,1],
    [1,0,0],
    [0,0,0]
]

print(shortest_path_maze(maze, (0,0), (2,2)))

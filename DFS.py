# DFS Practice Problem 1:
# Count connected components in an undirected graph

def count_components(n, edges):
    # Step 1: build adjacency list
    graph = {i: [] for i in range(n)}
    for a, b in edges:
        graph[a].append(b)
        graph[b].append(a)

    visited = set()

    # Step 2: DFS
    def dfs(node):
        visited.add(node)
        for nei in graph[node]:
            if nei not in visited:
                dfs(nei)

    # Step 3: count components
    components = 0
    for node in range(n):
        if node not in visited:
            dfs(node)
            components += 1

    return components


# ------------ TEST ------------
n = 5
edges = [(0, 1), (1, 2), (3, 4)]
print("Connected Components:", count_components(n, edges))


# DFS Practice Problem 2:
# Number of Islands in a grid

def num_islands(grid):
    if not grid:
        return 0

    rows = len(grid)
    cols = len(grid[0])
    visited = set()

    def dfs(r, c):
        # stop if out of bounds
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return

        # stop if water or visited
        if grid[r][c] == "0" or (r, c) in visited:
            return

        # mark as visited
        visited.add((r, c))

        # explore 4 directions
        dfs(r+1, c)  # down
        dfs(r-1, c)  # up
        dfs(r, c+1)  # right
        dfs(r, c-1)  # left

    islands = 0

    # scan entire grid
    for r in range(rows):
        for c in range(cols):
            # if we find new land, run DFS
            if grid[r][c] == "1" and (r, c) not in visited:
                dfs(r, c)
                islands += 1

    return islands


# ------------ TEST ------------
grid = [
    ["1","1","0","0"],
    ["1","0","0","1"],
    ["0","0","1","1"]
]
print("Number of Islands:", num_islands(grid))


# DFS Practice Problem 3:
# Check if a path exists in a maze (0=open, 1=wall)

def path_exists(maze, start, end):
    rows = len(maze)
    cols = len(maze[0])

    visited = set()

    def dfs(r, c):
        # TODO: base cases + DFS movement
        pass

    return dfs(start[0], start[1])


# ------------ TEST ------------
maze = [
    [0,0,1],
    [1,0,1],
    [0,0,0]
]

start = (0,0)
end = (2,2)

print("Path exists:", path_exists(maze, start, end))



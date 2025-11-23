# DFS Practice Problem 1:
# Count connected components in an undirected graph

def count_components(n, edges):
    # TODO: build adjacency list

    visited = set()

    def dfs(node):
        # TODO: implement DFS
        pass

    components = 0

    # TODO: loop through nodes and count components
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

    def dfs(r, c):
        # TODO: stop if out of bounds or water or visited
        pass

    islands = 0

    # TODO: loop through grid and call DFS
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



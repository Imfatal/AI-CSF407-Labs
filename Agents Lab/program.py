from collections import deque

# Warehouse map from the laboratory sheet
warehouse = [
    "#####################",
    "#S....#............G#",
    "#.##....##########..#",
    "#....##.............#",
    "#.######.###.#.###..#",
    "#........#..........#",
    "#####################"
]

# Movement directions:
# Up, Down, Left, Right
directions = [
    (-1, 0),  # Up
    (1, 0),   # Down
    (0, -1),  # Left
    (0, 1)    # Right
]


def find_position(grid, symbol):
    """Find the position of a symbol in the warehouse."""
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == symbol:
                return (r, c)
    return None


def find_path(grid):
    """
    Find a collision-free path from S to G using BFS.
    Returns the path as a list of positions.
    """

    start = find_position(grid, "S")
    goal = find_position(grid, "G")

    if start is None or goal is None:
        return None

    # Queue stores paths
    queue = deque([[start]])

    # Keep track of visited positions
    visited = {start}

    while queue:

        path = queue.popleft()
        current = path[-1]

        # Goal reached
        if current == goal:
            return path

        r, c = current

        # Try all four possible actions
        for dr, dc in directions:

            new_r = r + dr
            new_c = c + dc

            # Check that the new position is inside the grid
            if not (0 <= new_r < len(grid)):
                continue

            if not (0 <= new_c < len(grid[0])):
                continue

            # Check that the cell is not an obstacle
            if grid[new_r][new_c] == "#":
                continue

            new_position = (new_r, new_c)

            # Avoid visiting the same position repeatedly
            if new_position in visited:
                continue

            visited.add(new_position)

            # Add the new position to the path
            new_path = path + [new_position]
            queue.append(new_path)

    # No path exists
    return None


def print_path(grid, path):
    """Print the warehouse with the discovered path."""

    if path is None:
        print("No path exists from S to G.")
        return

    path_set = set(path)

    for r, row in enumerate(grid):

        line = ""

        for c, cell in enumerate(row):

            if (r, c) in path_set and cell not in ("S", "G"):
                line += "*"
            else:
                line += cell

        print(line)


# Find the path
path = find_path(warehouse)

# Print the result
if path is not None:
    print("Path found!")
    print("Path length:", len(path) - 1)
    print("Path:")
    print(path)

    print("\nWarehouse with path:")
    print_path(warehouse, path)

else:
    print("No path exists from S to G.")
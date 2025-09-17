import heapq
import plotly.graph_objects as go


# --- A* Pathfinding ---
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(grid, start, goal):
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))

    while oheap:
        current = heapq.heappop(oheap)[1]
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        close_set.add(current)
        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)
            tentative_g_score = gscore[current] + 1
            if 0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0]):
                if grid[neighbor[0]][neighbor[1]] == 1:
                    continue
            else:
                continue
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, float("inf")):
                continue
            if tentative_g_score < gscore.get(neighbor, float("inf")):
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    return []


# --- Grid & Path ---
grid = [
    [0, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 1, 0],
    [1, 0, 1, 0, 1, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
]

start, goal = (0, 0), (4, 5)
path = astar(grid, start, goal)

# --- Visualization ---
frames = []
visited_colors = [['white' if grid[i][j] == 0 else 'black' for j in range(len(grid[0]))] for i in range(len(grid))]

# Initial plot
fig = go.Figure()

# Add initial heatmap (grid)
fig.add_trace(go.Heatmap(
    z=[[1 if c == 'black' else 0 for c in row] for row in visited_colors],
    colorscale=[(0, 'white'), (1, 'black')],
    showscale=False
))

# Add agent position
fig.add_trace(go.Scatter(
    x=[start[1]], y=[start[0]],
    mode="markers", marker=dict(color="red", size=15),
    name="Agent"
))

# Create animation frames
for step, (x, y) in enumerate(path):
    # Copy colors for each step
    step_colors = [['white' if grid[i][j] == 0 else 'black' for j in range(len(grid[0]))] for i in range(len(grid))]

    # Mark visited path
    for (vx, vy) in path[:step + 1]:
        step_colors[vx][vy] = 'lightblue'

    frames.append(go.Frame(
        data=[
            go.Heatmap(
                z=[[1 if c == 'black' else (0.5 if c == 'lightblue' else 0) for c in row] for row in step_colors],
                colorscale=[(0, 'white'), (0.5, 'lightblue'), (1, 'black')],
                showscale=False
            ),
            go.Scatter(
                x=[y], y=[x],
                mode="markers", marker=dict(color="red", size=15),
                name="Agent"
            )
        ],
        name=str(step)
    ))

# Update figure layout for animation
fig.update(frames=frames)
fig.update_layout(
    title="A* Pathfinding with Moving Agent & Trail",
    xaxis=dict(scaleanchor="y", showgrid=True, zeroline=False, range=[-0.5, len(grid[0]) - 0.5]),
    yaxis=dict(autorange="reversed", showgrid=True, zeroline=False, range=[-0.5, len(grid) - 0.5]),
    updatemenus=[dict(
        type="buttons",
        buttons=[
            dict(label="Play", method="animate",
                 args=[None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]),
            dict(label="Pause", method="animate",
                 args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
        ]
    )]
)

fig.show()

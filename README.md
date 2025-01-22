# Cell Evolution Game
![image](https://github.com/user-attachments/assets/476f8716-61ee-4251-ab3a-ca3bfa9abb35)

## Overview
Cell Evolution Game is a simulation game featuring cells and predators. Cells divide at regular intervals, and predators eat the cells. Cells have characteristics such as size, speed, and color through genes, and genes can mutate during division.

## Game Features
- **Cell Division**: Cells divide at regular intervals to create new cells.
- **Predators**: Predators eat cells and reflect away when they approach the safe zone.
- **Gene Mutation**: Cells' genes mutate during division, resulting in various characteristics.
- **Mouse Hovering**: Hovering the mouse over a cell displays its information in a tooltip.
- **HUD**: Displays game time, FPS, cell count, predator count, and maximum generation in the HUD.
- **Graph**: Visualizes the change in cell count with a graph.

## Installation and Execution
1. **Install Required Libraries**:
    ```bash
    pip install pygame
    ```

2. **Run the Game**:
    ```bash
    python main.py
    ```

## Controls
- **Mouse**: Hover over a cell to see its information in a tooltip.
- **Exit**: Close the game window to exit the game.

## Code Structure
- `main.py`: Contains the main logic of the game.
- `Particle` Class: Manages particle effects.
- `Cell` Class: Manages cell objects.
- `Predator` Class: Manages predator objects.

## Key Variables
- `WIDTH`, `HEIGHT`: Game screen dimensions.
- `FPS`: Frames per second.
- `MAX_CELLS`: Maximum number of cells.
- `MAX_PREDATORS`: Maximum number of predators.
- `SAFE_ZONE_RADIUS`: Radius of the safe zone.
- `SPLIT_INTERVAL`: Cell division interval (seconds).
- `PREDATOR_SPAWN_DELAY`: Predator spawn interval (seconds).
- `PREDATOR_MAX_SPEED`: Maximum speed of predators.

## Game Screen
- **Background**: Dark-colored background.
- **Cells**: Cells with various colors and sizes.
- **Predators**: White square-shaped predators.
- **Safe Zone**: Circular safe zone located in the center.
- **HUD**: Displays game information at the top of the screen.
- **Graph**: Displays the change in cell count at the top right of the screen.

## Contribution
If you want to contribute to this project, please file an issue or submit a pull request.

# Othello-Game
Othello featuring AI powered by Mini-Max with alpha-beta pruning

## Overview
This project implements the classical two-player game **Othello**, along with an AI opponent powered by the **Mini-Max heuristic search algorithm**. The AI includes **alpha-beta pruning** for optimization and a configurable heuristic for decision-making. 

---

## Features

### Game Mechanics
- Complete implementation of Othello rules:
  - Board representation and initialization.
  - Enforcement of legal moves and flipping discs.
  - ASCII-based board visualization after each move.
  - Scorekeeping and detection of game completion.
- Two-player mode, allowing humans to play Othello against each other.

### AI Opponent
- **Mini-Max Algorithm**:
  - Debug mode to display sequences of moves considered and their heuristic values.
  - Adjustable search depth for experimentation and optimization.
  - Tracks the total number of game states examined per move.
- **Alpha-Beta Pruning**:
  - Toggleable pruning to compare performance with and without optimization.
  - Demonstrates reduced game state evaluations when enabled.
- AI can play as either **black** or **white**, allowing the human player to choose their preferred side.

---

## Requirements
- Language: Python
- A terminal or command-line environment for running the game.

---

## Instructions
1. Run the game from its directory: python othello.py

#################################
# Name: Drew Ferrington
# CSC 475: Othello
# Created: Nov. 11, 2024
#################################

"""
~DREW'S DEV NOTES~

Othello Game

Description:
This program implements the game "Othello" for two human players or a human player against an AI opponent.
The AI uses the minimax algorithm with alpha-beta pruning and a heuristic evaluation function. Players can 
choose to have the AI make moves for them on a turn-by-turn basis, and they can adjust the search depth at which
the AI will look to find optimal plays against an opponent. The game is played in the command line with ASCII graphics.

Features:
- Two-player mode or play against the AI.
- Players choose their colors in two-player mode.
- Players can have the AI make moves for them on a turn-by-turn basis.
- Minimax algorithm with adjustable search depth.
- Alpha-beta pruning that can be enabled or disabled on a move-by-move basis.
- Debug mode to display the AI's thought process.
- Ability to adjust the AI's search depth on a turn-by-turn basis, with a maximum depth limit.
- Scorekeeping and game completion recognition.

Instructions:
- Run program and follow prompts to play the game.
- Choose your color (Black or White) or select two-player mode.
- Set initial AI search depth (maximum allowed is 8) when playing against the AI.
- On each turn, you can choose to have the AI make a move for you.
- When the AI is making a move, you can adjust the search depth and enable/disable alpha-beta pruning and debug mode.
- The game ends when neither player can make a valid move.
"""

import copy  # importing copy module to create deep copies of the board during simulations
# global variables for debug mode and nodes examined
DEBUG_MODE = False  # debug mode is off by default
nodes_examined = 0  # counter for the number of nodes examined during minimax
MAX_DEPTH = 8  # maximum allowed search depth

#############################################~FUNCTIONS START HERE~###################################################

def display_board(board):
    """
    This function displays current state of game board along with current scores for both players.
    It prints the board in a readable format with row and column indices, and shows the count
    of black and white discs; visualizes game state after each move; called by main() at start and
    at end of each turn to show updates
    """
     # calculate current scores
    black_score = sum(row.count('B') for row in board)  # count black discs
    white_score = sum(row.count('W') for row in board)  # count white discs
    # display scores
    print() # spacing
    print(f"Current Scores - Black: {black_score}, White: {white_score}")
    print()
    print('  ' + ' '.join([str(i) for i in range(8)]))  # print column headers (0-7)
    for idx, row in enumerate(board):  # iterate over each row with its index
        print(f"{idx} " + ' '.join(row))  # print row number and contents
    print()  # spacing


"""
This function prompts player to enter their move in the format 'row,col'; validates input,
ensuring it's in correct format and is one of the valid moves provided; called in main() when player chooses their own move
"""
def get_player_move(valid_moves):
    while True:
        try:
            move = input("Enter your move as 'row,col': ")  # prompt user for input
            x_str, y_str = move.strip().split(',')  # split input into row and column
            x, y = int(x_str), int(y_str)  # convert to integers
            if (x, y) in valid_moves:  # check if move is valid
                return x, y  # return valid move
            else:
                print("Invalid move. Please choose from the available moves.")  # invalid move
        except ValueError:
            print("Invalid input format. Please enter row and column separated by a comma.")  # input format error


"""
This function calculates/returns list of valid moves for specified player based on current
state of board; checks each empty cell on board, determining if placing a disc there
would capture any of the opponent's discs according to game rules; involves checking
in all eight directions from each empty cell; called in main() at beginning of each turn,
minimax() to generate valid moves at each node in game tree during AI simulations, and ai_move() to
find all possible moves for AI's decision to move
"""
def get_valid_moves(board, player):
    valid_moves = []  # initialize list of valid moves
    opponent = 'W' if player == 'B' else 'B' # determine opponent's color
    # Define all eight possible directions; each direction is represented as 
    # a tuple (dx, dy) indicating the change in the x (row) and y (column) coordinates respectively.
    directions = [(-1, -1), (-1, 0), (-1, 1), # up-left, up, up-right
                  (0, -1),         (0, 1),    # left,        right
                  (1, -1),  (1, 0), (1, 1)] # down-left, down, down-right
    for x in range(8):  # loop over rows
        for y in range(8):  # loop over columns
            if board[x][y] != ' ':  # skip occupied cells
                continue
            for dx, dy in directions:  # check all directions from current cell
                nx, ny = x + dx, y + dy  # calculate next cell in the direction
                found_opponent = False  # flag to indicate if opponent's disc is found
                while 0 <= nx < 8 and 0 <= ny < 8 and board[nx][ny] == opponent:
                    nx += dx  # move to next cell in same direction
                    ny += dy
                    found_opponent = True  # opponent's disc found, set flag
                if found_opponent and 0 <= nx < 8 and 0 <= ny < 8 and board[nx][ny] == player: # after finding opponent disc, check if we end up on player disc
                    valid_moves.append((x, y))  # valid move found, so add original empty cell
                    break  # stop checking other directions for this cell
    return valid_moves  # return list of valid moves


"""
This function executes a move for the specified player by placing their disc at given coordinates (x, y)
and flipping opponent's discs as per game rules. Updates board state accordingly, affecting all lines 
(in all eight directions) where player's disc brackets opponent's discs, progressing game state after each move;
used in main() to execute player/AI moves, in minimax() and ai_move() to simulate moves on copied boards during AI recursive exploration,
""" 
def make_move(board, player, x, y):
    board[x][y] = player  # place player's disc on chosen cell
    opponent = 'W' if player == 'B' else 'B'  # determine opponent's color
    # Define all eight possible directions similar to get_valid_moves
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),         (0, 1),
                  (1, -1),  (1, 0), (1, 1)]
    for dx, dy in directions:  # Check all directions for flippable discs
        discs_to_flip = []  # list holds opponent's discs to flip
        nx, ny = x + dx, y + dy  # next position in direction; start checking from adjacent cell
        while 0 <= nx < 8 and 0 <= ny < 8 and board[nx][ny] == opponent: # move in direction while we find opponent discs
            discs_to_flip.append((nx, ny))  # Add position to flip list
            nx += dx  # move to next cell, same direction
            ny += dy
        if discs_to_flip and 0 <= nx < 8 and 0 <= ny < 8 and board[nx][ny] == player: # after collecting opponent's discs, check if we end up on player disc
            for fx, fy in discs_to_flip:
                board[fx][fy] = player  # flip all the collected discs to player's color


"""
This function evaluates board from AI player's perspective, determining how favorable current
state is; calculates score based on difference in disc counts between AI player
and opponent, and adds weighted value for corner positions, which are strategically
advantageous. This heuristic evaluation is used by minimax algorithm to make decisions
without searching to the end of the game; called in minimax() when max depth reached or no valid 
moves to evaluate heuristic score of board and let AI assess quality of game state
"""
def evaluate_board(board, ai_player):
    opponent = 'W' if ai_player == 'B' else 'B'  # determine opponent's color
    ai_count = sum(row.count(ai_player) for row in board)  # count AI's discs
    opponent_count = sum(row.count(opponent) for row in board)  # count opponent's discs
    score = ai_count - opponent_count  # basic score difference
    corners = [(0, 0), (0, 7), (7, 0), (7, 7)] # corner positions
    ai_corners = sum(1 for x, y in corners if board[x][y] == ai_player)  # AI's occupied corners
    opponent_corners = sum(1 for x, y in corners if board[x][y] == opponent)  # Opponent's occupied corners
    corner_score = 25 * (ai_corners - opponent_corners)  # assign higher weight to corner positions
    return score + corner_score  # Total evaluation score


"""
This function implements minimax algorithm with optional alpha-beta pruning to determine best move
for AI player; recursively explores possible moves to a specified depth, evaluating board at leaf nodes. 
The 'maximizingPlayer' parameter indicates whether current recursion level is maximizing or minimizing the evaluation score;
'alpha' is best value the maximizer can guarantee at that level or above, and 'beta' is same but for minimizer (floats);'indent' parameter is used for formatting debug 
output to visualize recursion depth. Alpha-beta pruning eliminates branches that cannot yield better results; called in ai_move() to evaluate
potential moves AI can make, and in minimax() where it recursively calls itself to keep exploring game tree
"""
def minimax(board, depth, alpha, beta, maximizingPlayer, ai_player, use_pruning=True, indent=""):
    global nodes_examined  # access global counter
    nodes_examined += 1  # increment nodes examined by AI as it is called recursively for each possible move
    opponent = 'W' if ai_player == 'B' else 'B' # determine opponent's color
    current_player = ai_player if maximizingPlayer else opponent  # set current player
    valid_moves = get_valid_moves(board, current_player) # get valid moves
    if depth == 0 or not valid_moves: # evaluate base case/leaf node
        eval_score = evaluate_board(board, ai_player) # evaluate the board
        if DEBUG_MODE:
            print(f"{indent}Evaluated leaf node: score {eval_score}") # debug output
        return eval_score # return evaluation score
    # MAXIMIZING PLAYER'S LOGIC
    if maximizingPlayer:
        max_eval = float('-inf') # initialize maximum evaluation to negative infinity
        if DEBUG_MODE:
            print(f"{indent}Maximizing player at depth {depth}")
        for x, y in valid_moves:
            new_board = copy.deepcopy(board)  # create deep copy of board to simulate the move
            make_move(new_board, ai_player, x, y) # simulate move on new board
            if DEBUG_MODE:
                print(f"{indent}Trying move ({x}, {y})")
            # recursive call to minimax for the minimizer (opponent) increase the indentation to reflect deeper recursion
            eval = minimax(new_board, depth - 1, alpha, beta, False, ai_player, use_pruning, indent + "    ")
            # indentation increases here (forward indentation) as we go deeper into recursion
            max_eval = max(max_eval, eval)  # update maximum evaluation
            alpha = max(alpha, eval)  # UPDATE ALPHA for alpha-beta pruning
            if DEBUG_MODE:
                print(f"{indent}Move ({x}, {y}) resulted in eval {eval}, current max_eval {max_eval}")
            # alpha-beta pruning condition; if best score minimizer can guarantee (beta) no need to keep exploring
            if use_pruning and beta <= alpha:
                if DEBUG_MODE:
                    print(f"{indent}Pruning remaining moves at depth {depth} (alpha >= beta)")
                break # prune remaining branches
            # when recursive call returns, indentation decreases (backward indentation) as we backtrack to the previous recursion level
        return max_eval # return maximum evaluation to ai_move() where best move can be returned to main()
    # MINIMIZING PLAYER'S LOGIC
    else:
        min_eval = float('inf') # initialize minimum evaluation to positive infinity
        if DEBUG_MODE:
            print(f"{indent}Minimizing player at depth {depth}")
        for x, y in valid_moves:
            new_board = copy.deepcopy(board) # create a deep copy of the board to simulate the move
            make_move(new_board, opponent, x, y) # simulate the move on the new board
            if DEBUG_MODE:
                print(f"{indent}Trying move ({x}, {y})")
            # recursive call to minimax for the maximizer (AI player) increase the indentation to reflect deeper recursion
            eval = minimax(new_board, depth - 1, alpha, beta, True, ai_player, use_pruning, indent + "    ")
            # indentation increases here (forward indentation) as we go deeper into recursion
            min_eval = min(min_eval, eval)  # Update the minimum evaluation
            beta = min(beta, eval)  # UPDATE BETA for alpha-beta pruning
            if DEBUG_MODE:
                print(f"{indent}Move ({x}, {y}) resulted in eval {eval}, current min_eval {min_eval}")
            # alpha-beta pruning condition; if best score maximizer can guarantee (alpha) no need to keep exploring
            if use_pruning and beta <= alpha:
                if DEBUG_MODE:
                    print(f"{indent}Pruning remaining moves at depth {depth} (alpha >= beta)")
                break  # prune remaining branches
            # when recursive call returns, indentation decreases (backward indentation) as we backtrack to previous recursion level
        return min_eval # return minimum evaluation to ai_move() where best move can be returned to main()


"""
This function initializes alpha/beta and determines AI's best move using minimax algorithm; called in main() when AI's turn or player wants AI to move for them
"""
def ai_move(board, depth, ai_player, use_pruning=True):
    global nodes_examined # access global counter for nodes examined
    nodes_examined = 0 # reset nodes_examined counter for this move
    valid_moves = get_valid_moves(board, ai_player) # get valid moves for AI player
    if not valid_moves:
        return None # no valid moves
    best_move = None # initialize best move
    max_eval = float('-inf')  # initialize max evaluation to neg inf
    alpha = float('-inf') # initialize alpha for alpha-beta pruning
    beta = float('inf') # initialize beta for alpha-beta pruning
    for x, y in valid_moves: # loop through all valid moves to find best one
        new_board = copy.deepcopy(board)  # create a deep copy of the board for move simulation
        make_move(new_board, ai_player, x, y)  # simulate move
        eval = minimax(new_board, depth - 1, alpha, beta, False, ai_player, use_pruning)  # evaluate move
        if eval > max_eval:
            max_eval = eval # update max evaluation
            best_move = (x, y) # update best move
        alpha = max(alpha, eval) # update alpha
        if DEBUG_MODE:
            print(f"AI considering move ({x}, {y}) with eval {eval}") # debug output
    if DEBUG_MODE:
        print(f"AI evaluated {nodes_examined} nodes.") # debug output
    return best_move # return best move


"""
MAIN: setup and manage game loop and player inputs
"""
def main():
    global DEBUG_MODE # declare global DEBUG_MODE at beginning
    board = [[' ' for _ in range(8)] for _ in range(8)] # initialize the game board with empty spaces aka two-dimensional list
    # set up initial positions (four discs in the center)
    board[3][3], board[3][4] = 'W', 'B' 
    board[4][3], board[4][4] = 'B', 'W'
    current_player = 'B'  # according to rules of othello, black starts first
    # prompt the user to select the game mode
    print()
    print("~ WELCOME TO OTHELLO! ~")
    print()
    print("Select game mode:")
    print("1. Two-player mode")
    print("2. Play against AI")
    print()
    mode = input("Enter 1 or 2: ").strip()
    while mode not in ['1', '2']: # validate input
        mode = input("Invalid selection. Please enter 1 or 2: ").strip()
    two_player_mode = (mode == '1') # determine game mode
    if two_player_mode:
        # TWO PLAYER MODE, prompt players to choose their colors
        print("Player 1, choose your color:")
        player1_color = input("Do you want to be Black (B) or White (W)? ").strip().upper()
        while player1_color not in ['B', 'W']:
            player1_color = input("Invalid choice. Please enter 'B' for Black or 'W' for White: ").strip().upper()
        player2_color = 'W' if player1_color == 'B' else 'B'  # assign other color to Player 2
        print(f"Player 1 is {player1_color}. Player 2 is {player2_color}.")
    else:
        # PLAY AGAINST AI
        player_color = input("Do you want to be Black (B) or White (W)? ").strip().upper()
        while player_color not in ['B', 'W']:
            player_color = input("Invalid choice. Please enter 'B' for Black or 'W' for White: ").strip().upper()
        #ai_player = 'W' if player_color == 'B' else 'B' (# assign AI's color) THIS WAS A REDUNDANCY ACCIDENTALLY LEFT IN SUBMITTED VERSION BECAUSE I FORGOT TO REMOVE BUT IT DOES NOT AFFECT ANYTHING, current_player alternates between B and W and when current_player equals player_color its humans turn and vice versa, so current player is determining AI color
        # set initial depth with validation
        while True:
            try:
                depth = int(input(f"Enter an initial AI search depth from 1 to {MAX_DEPTH} (3 for typical performance): ").strip())
                if 1 <= depth <= MAX_DEPTH:
                    break
                else:
                    print(f"Depth must be between 1 and {MAX_DEPTH}.")
            except ValueError:
                print("Invalid input. Please enter an integer.")
    display_board(board)  # display initial board after getting player colors and depth info
    while True: # main game loop
        valid_moves = get_valid_moves(board, current_player) # get valid moves
        if not valid_moves:
            # check if opponent has moves
            opponent = 'W' if current_player == 'B' else 'B' # determine opponent
            if not get_valid_moves(board, opponent):
                break # game over if no one can move
            else:
                print(f"{current_player} has no valid moves. Turn is passed.")
                current_player = opponent # pass the turn
                continue
        print(f"Current Player: {current_player}")
        if two_player_mode:
            # TWO PLAYER MODE
            if current_player == player1_color:
                print("Player 1's turn.")
            else:
                print("Player 2's turn.")
            # ask if the player wants the AI to make a move for them
            while True:
                ai_for_player = input("Do you want the AI to make a move for you? (y/n): ").strip().lower()
                if ai_for_player in ['y', 'n']:
                    break
                else:
                    print("Invalid input. Please enter 'y' or 'n'.")
            # PLAYER SAYS YES MAKE A MOVE FOR ME
            if ai_for_player == 'y':
                # ask the player to set AI parameters for this move
                while True:
                    try:
                        depth = int(input(f"Enter the AI search depth for this move (1 to {MAX_DEPTH}): ").strip())
                        if 1 <= depth <= MAX_DEPTH:
                            break
                        else:
                            print(f"Depth must be between 1 and {MAX_DEPTH}.")
                    except ValueError:
                        print("Invalid input. Please enter an integer.")
                # ask player for pruning selection
                while True:
                    pruning_input = input("Enable alpha-beta pruning for this move? (y/n): ").strip().lower()
                    if pruning_input in ['y', 'n']:
                        use_pruning = pruning_input == 'y' # set pruning option
                        break
                    else:
                        print("Invalid input. Please enter 'y' or 'n'.")
                # ask to enable debug mode for AI's move
                while True:
                    debug_input = input("Enable debug mode for AI's move? (y/n): ").strip().lower()
                    if debug_input in ['y', 'n']:
                        DEBUG_MODE = debug_input == 'y' # set debug mode
                        break
                    else:
                        print("Invalid input. Please enter 'y' or 'n'.")
                # AI makes a move for the player
                move = ai_move(board, depth=depth, ai_player=current_player, use_pruning=use_pruning)
                if move:
                    make_move(board, current_player, *move)  # execute move
                    print(f"AI chose move {move} for you.")
                    print(f"AI evaluated {nodes_examined} nodes.")
                else:
                    print("No valid moves available.")
            # PLAYER SAYS NO I'LL MAKE MY OWN MOVE
            else:
                # get move from the current player
                print("Valid moves:", valid_moves)
                x, y = get_player_move(valid_moves)  # get player's move
                make_move(board, current_player, x, y) # execute move
        # HUMAN VS AI
        else:
            if current_player == player_color:
                print("Your turn!")  # Human player's turn
                # ask if player wants AI to make move for them
                while True:
                    ai_for_player = input("Do you want the AI to make a move for you? (y/n): ").strip().lower()
                    if ai_for_player in ['y', 'n']:
                        break
                    else:
                        print("Invalid input. Please enter 'y' or 'n'.")
                # PLAYER SAYS YES MAKE A MOVE FOR ME
                if ai_for_player == 'y':
                    # ask player to set AI parameters for this move
                    while True:
                        try:
                            depth = int(input(f"Enter the AI search depth for this move (1 to {MAX_DEPTH}): ").strip())
                            if 1 <= depth <= MAX_DEPTH:
                                break
                            else:
                                print(f"Depth must be between 1 and {MAX_DEPTH}.")
                        except ValueError:
                            print("Invalid input. Please enter an integer.")
                    # ask player to make pruning selection
                    while True:
                        pruning_input = input("Enable alpha-beta pruning for this move? (y/n): ").strip().lower()
                        if pruning_input in ['y', 'n']:
                            use_pruning = pruning_input == 'y' # set pruning option
                            break
                        else:
                            print("Invalid input. Please enter 'y' or 'n'.")
                    # ask player to enable debug mode for AI's move
                    while True:
                        debug_input = input("Enable debug mode for AI's move? (y/n): ").strip().lower()
                        if debug_input in ['y', 'n']:
                            DEBUG_MODE = debug_input == 'y' # set debug mode
                            break
                        else:
                            print("Invalid input. Please enter 'y' or 'n'.")
                    # AI makes a move for the player
                    move = ai_move(board, depth=depth, ai_player=current_player, use_pruning=use_pruning)
                    if move:
                        make_move(board, current_player, *move) # execute the move
                        print(f"AI chose move {move} for you.")
                        print(f"AI evaluated {nodes_examined} nodes.")
                    else:
                        print("No valid moves available.")
                # PLAYER SAYS NO I'LL MAKE MY OWN MOVE
                else:
                    # player makes their own move
                    print("Valid moves:", valid_moves)
                    x, y = get_player_move(valid_moves) # get the player's move
                    make_move(board, current_player, x, y) # execute the move
            # AI'S TURN
            else:
                print("AI is thinking...")
                print(f"Current AI search depth: {depth}")
                # ask if the player wants to adjust AI's search depth
                while True:
                    adjust_depth_input = input("Would you like to adjust the AI search depth for this move? (y/n): ").strip().lower()
                    if adjust_depth_input in ['y', 'n']:
                        break
                    else:
                        print("Invalid input. Please enter 'y' or 'n'.")
                if adjust_depth_input == 'y':
                    # adjust depth
                    try:
                        new_depth = int(input(f"Enter the new AI search depth (1 to {MAX_DEPTH}): ").strip())
                        if new_depth < 1 or new_depth > MAX_DEPTH:
                            print(f"Depth must be between 1 and {MAX_DEPTH}. Keeping the previous depth.")
                        else:
                            depth = new_depth # update depth
                    except ValueError:
                        print("Invalid input. Keeping the previous depth.")
                # ask player if they want to enable debug mode for AI's move
                debug_input = input("Enable debug mode for AI's move? (y/n): ").strip().lower()
                DEBUG_MODE = debug_input == 'y'
                pruning_input = input("Enable alpha-beta pruning for this move? (y/n): ").strip().lower()
                use_pruning = pruning_input == 'y' # set pruning option
                move = ai_move(board, depth=depth, ai_player=current_player, use_pruning=use_pruning)
                # AI MAKES ITS MOVE
                if move:
                    make_move(board, current_player, *move) # execute move
                    print(f"AI places at {move}")
                    if not DEBUG_MODE:
                        print(f"AI evaluated {nodes_examined} nodes.")
                else:
                    print("AI has no valid moves.")
        display_board(board) # display the board after each turn
        current_player = 'W' if current_player == 'B' else 'B' # switch player
    # game over, calculate final scores
    black_score = sum(row.count('B') for row in board) # count black discs
    white_score = sum(row.count('W') for row in board) # count white discs
    print(f"GAME OVER! Final Scores:\nBlack: {black_score}, White: {white_score}")  # display scores
    if black_score > white_score:
        print("BLACK WINS!") # Black wins
    elif white_score > black_score:
        print("WHITE WINS!") # White wins
    else:
        print("It's a tie!") # Tie game

#############################################~FUNCTIONS END HERE~#######################################################

main() # start the game




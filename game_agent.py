"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
from isolation import Board
from copy import deepcopy
from copy import copy


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    mc_factor = game.move_count * 0.2
    weights = [1., -2.0, None, -0.5]
    total_score = 0.
    # own_move_score (rescaled from 0 to 1)
    if weights[0] is not None:
        own_move = float(len(game.get_legal_moves(player)))
        total_score += weights[0] * own_move / 8.
    # opp_move_score (rescaled from 0 to 1)
    if weights[1] is not None:
        opp_move = float(len(game.get_legal_moves(game.get_opponent(player))))
        total_score += weights[1] * opp_move / 8.
    # fight_or_flight (rescaled from 0 to 1)
    if weights[2] is not None:
        own_loc = game.get_player_location(player)
        opp_loc = game.get_player_location(game.get_opponent(player))
        h1, w1 = own_loc
        h2, w2 = opp_loc
        jump_distance = float(player.knight_distance[h1 - h2][w1 - w2])
        total_score += weights[2] * jump_distance / 5. / mc_factor
    # control_center_score (rescaled from 0 to 1)
    if weights[3] is not None:
        own_loc = game.get_player_location(player)
        h1, w1 = own_loc
        control = float(player.board_position_score[h1][w1])
        total_score += weights[3] * control / 8. / mc_factor

    return total_score


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=2.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

        # list of possible knight movements
        self.coord_moves = [(-2, -1), (-2, 1), (2, -1), (2, 1),
                            (-1, -2), (-1, 2), (1, -2), (1, 2)]
        # assumed board dimensions for precomputations
        board_width = 7
        board_height = 7

        # score board positions based on potential knight movements
        self.board_position_score = [[0 for i in range(board_width)] for j
                                     in range(board_height)]
        for i in range(7):
            for j in range(7):
                valid_moves = 0
                for c in self.coord_moves:
                    if i + c[0] >= 0 and i + c[0] <= 6:
                        if j + c[1] >= 0 and j + c[1] <= 6:
                            valid_moves += 1
                self.board_position_score[i][j] = valid_moves

        # create matrix to get number of jumps from opponent (knight distance)
        def fill_knight_distance(seed_dist, graph, spots):
            new_spots = []
            for i, j in spots:
                moves = [(i + m[0], j + m[1]) for m in self.coord_moves]
                for a, b in moves:
                    if a >= -6 and a <= 6 and b >= -6 and b <= 6:
                        if graph[a][b] is None:
                            graph[a][b] = seed_dist + 1
                            new_spots.append((a, b))
            if new_spots:
                fill_knight_distance(seed_dist + 1, graph, new_spots)

        rows = range(-board_width + 1, board_width)
        cols = range(-board_width + 1, board_height)
        self.knight_distance = [[None for i in rows] for j in cols]
        self.knight_distance[0][0] = 0
        fill_knight_distance(0, self.knight_distance, [(0, 0)])

        # Extras
        self.my_score = None
        self.explored_depth = None
        self.explored_nodes = None

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        if not legal_moves:
            return (-1, -1)

        # Own board implementation to avoid computation intensive forecasting
        game = EfficientBoard(game)

        try:
            if self.method == 'minimax':
                method = self.minimax
            elif self.method == 'alphabeta':
                method = self.alphabeta
            else:
                raise Exception('Method %s not implemented' % self.method)

            # Apply iterative deepening or or limit search by search_depth
            if self.iterative:
                depth = 1
                self.explored_depth = 0
                self.explored_nodes = 0
                while True:
                    self.my_score, self.my_move = method(game, depth)
                    self.explored_depth = depth
                    if (self.my_score == float('inf') or
                            self.my_score == float('-inf')):
                        return self.my_move
                    depth += 1
            else:
                self.explored_depth = self.search_depth
                self.explored_nodes = 0
                self.my_score, self.my_move = method(game, self.search_depth)
                return self.my_move

        except Timeout:
            # Handle any actions required at timeout, if necessary
            return self.my_move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Work with generators to optimize memory management (driven by depth)

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """
        # We get the legal moves for the current playing player
        legal_moves = game.get_legal_moves()

        # The player with no available moves lost (encountered terminal node)
        if not legal_moves:
            if maximizing_player:
                score = float('-inf')
            else:
                score = float('inf')
            return score, (-1, -1)

        # Initialize score and move values for the algorithm
        if maximizing_player:
            best_score = float('-inf')
            best_move = legal_moves[0]
        else:
            best_score = float('inf')
            best_move = legal_moves[0]

        # each move is a new node to explore in the tree search
        for move in legal_moves:
            if self.time_left() < self.TIMER_THRESHOLD:
                raise Timeout()
            self.explored_nodes += 1

            # apply move to explore next node
            game.apply_move(move)

            # As long as we dont reach the last desired depth we expand
            # the search tree on depth first basis
            if depth > 1:
                score, next_move = self.minimax(
                    game, depth - 1, not maximizing_player)
            else:
                score = self.score(game, self)

            # score optimization
            if maximizing_player:
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                if score < best_score:
                    best_score = score
                    best_move = move

            # Revert the game to its previous state to continue the search
            game.undo_move()

        return best_score, best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"),
                  maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """
        # We get the legal moves for the current playing player
        legal_moves = game.get_legal_moves()

        # The player with no available moves lost (encountered terminal node)
        if not legal_moves:
            if maximizing_player:
                score = float('-inf')
            else:
                score = float('inf')
            return score, (-1, -1)

        # Initialize score and move values for the algorithm
        if maximizing_player:
            best_score = float('-inf')
            best_move = legal_moves[0]
        else:
            best_score = float('inf')
            best_move = legal_moves[0]

        # each move is a new node to explore in the tree search
        for move in legal_moves:
            if self.time_left() < self.TIMER_THRESHOLD:
                raise Timeout()
            self.explored_nodes += 1

            # apply move to explore next node
            game.apply_move(move)

            # As long as we dont reach the last desired depth we expand
            # the search tree on depth first basis
            if depth > 1:
                score, _ = self.alphabeta(
                    game, depth - 1, alpha, beta, not maximizing_player)
            else:
                score = self.score(game, self)

            # score optimization
            if maximizing_player:
                if score > best_score:
                    best_score = score
                    best_move = move
                # Alpha-beta pruning
                if best_score >= beta:
                    game.undo_move()
                    return best_score, best_move
                alpha = max(alpha, best_score)
            else:
                if score < best_score:
                    best_score = score
                    best_move = move
                # Alpha-beta pruning
                if best_score <= alpha:
                    game.undo_move()
                    return best_score, best_move
                beta = min(beta, best_score)

            # Revert the game to its previous state to continue the search
            game.undo_move()

        return best_score, best_move


class EfficientBoard(Board):

    def __init__(self, board):
        Board.__init__(self, board.__player_1__, board.__player_2__,
                       board.width, board.height)
        self.move_count = board.move_count
        self.__active_player__ = board.__active_player__
        self.__inactive_player__ = board.__inactive_player__
        self.__last_player_move__ = copy(board.__last_player_move__)
        self.__player_symbols__ = copy(board.__player_symbols__)
        self.__board_state__ = deepcopy(board.__board_state__)
        # We also add a move history to tackle reversions and access the
        # history if desired from the board
        self.move_history = [self.__last_player_move__[self.__active_player__],
                             self.__last_player_move__[self.__inactive_player__]]

    def apply_move(self, move):
        """
        Move the active player to a specified location.

        Parameters
        ----------
        move : (int, int)
            A coordinate pair (row, column) indicating the next position for
            the active player on the board.

        Returns
        ----------
        None
        """
        row, col = move
        self.__last_player_move__[self.active_player] = move
        self.__board_state__[row][col] = self.__player_symbols__[self.active_player]
        self.__active_player__, self.__inactive_player__ = self.__inactive_player__, self.__active_player__
        self.move_count += 1
        # We also append move to move_history
        # only difference with original method
        self.move_history.append(move)

    def undo_move(self):
        """
        Undo last player move

        Parameters
        ----------

        Returns
        ----------
        None
        """
        # Empty position of the move
        current_move = self.move_history[-1]
        row, col = current_move
        self.__board_state__[row][col] = Board.BLANK

        # Assign player to its previous position
        previous_move = self.move_history[-3]
        self.__last_player_move__[self.inactive_player] = previous_move
        row, col = previous_move
        self.__board_state__[row][col] = self.__player_symbols__[self.inactive_player]

        # delete last traces of move
        self.__active_player__, self.__inactive_player__ = self.__inactive_player__, self.__active_player__
        self.move_count -= 1
        del self.move_history[-1]

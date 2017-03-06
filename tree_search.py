"""Test search algorithms on a simple tree"""
tree = [[[3, 5], [6, 9]], [[1, 2], [0, -1]]]


def alphabeta(tree, depth, alpha=float("-inf"), beta=float("inf"),
              maximizing_player=True):
    # We recursively search until reaching desired depth
    if depth > 1:
        search_results = (
            alphabeta(subtree, depth - 1, alpha,
                      beta, not maximizing_player) for subtree in tree)
    else:
        search_results = ((score, move) for
                          score, move in zip(tree, range(len(tree))))

    # optimize according to player and perform alpha/beta pruning
    if maximizing_player:
        best_score = float("-inf")
        for index, result in enumerate(search_results):
            if result[0] > best_score:
                best_score = result[0]
                best_move = index
            alpha = max(alpha, best_score)
            print(depth, result, alpha, beta)
            if best_score >= beta:
                print('pruned')
                return best_score, best_move
        return best_score, best_move
    else:
        best_score = float("inf")
        for index, result in enumerate(search_results):
            if result[0] < best_score:
                best_score = result[0]
                best_move = index
            beta = min(beta, best_score)
            print(depth, result, alpha, beta)
            if best_score <= alpha:
                print('pruned')
                return best_score, best_move
        return best_score, best_move


def minimax(tree, depth, maximizing_player=True):
    # We recursively search until reaching desired depth
    if depth > 1:
        search_results = (
            minimax(subtree, depth - 1, not maximizing_player) for
            subtree in tree)
    else:
        search_results = ((score, move) for
                          score, move in zip(tree, range(len(tree))))

    # optimize according to player and return score and move
    if maximizing_player:
        index, result = max(enumerate(search_results),
                            key=lambda x: x[1][0])
        print(depth, result)
    else:
        index, result = min(enumerate(search_results),
                            key=lambda x: x[1][0])
        print(depth, result)

    return result[0], index

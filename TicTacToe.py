import random

class TicTacToe:
    def __init__(self):
        self.board = [' ' for _ in range(9)]
        self.current_winner = None

    def display_board(self):
        for row in [self.board[i*3:(i+1)*3] for i in range(3)]:
            print('| ' + ' | '.join(row) + ' |')

    @staticmethod
    def display_board_numbers():
        number_board = [[str(i) for i in range(j*3, (j+1)*3)] for j in range(3)]
        for row in number_board:
            print('| ' + ' | '.join(row) + ' |')

    def available_moves(self):
        return [i for i, spot in enumerate(self.board) if spot == ' ']

    def has_empty_squares(self):
        return ' ' in self.board

    def count_empty_squares(self):
        return self.board.count(' ')

    def execute_move(self, square, letter):
        if self.board[square] == ' ':
            self.board[square] = letter
            if self.check_winner(square, letter):
                self.current_winner = letter
            return True
        return False

    def check_winner(self, square, letter):
        row_index = square // 3
        row = self.board[row_index*3:(row_index+1)*3]
        if all([spot == letter for spot in row]):
            return True
        
        col_index = square % 3
        column = [self.board[col_index+i*3] for i in range(3)]
        if all([spot == letter for spot in column]):
            return True

        if square % 2 == 0:
            diagonal1 = [self.board[i] for i in [0, 4, 8]]
            if all([spot == letter for spot in diagonal1]):
                return True
            diagonal2 = [self.board[i] for i in [2, 4, 6]]
            if all([spot == letter for spot in diagonal2]):
                return True
        return False


def play_game(game, x_player, o_player, display_game=True):
    if display_game:
        game.display_board_numbers()

    current_letter = 'X'
    while game.has_empty_squares():
        if current_letter == 'O':
            square = o_player.get_move(game)
        else:
            square = x_player.get_move(game)

        if game.execute_move(square, current_letter):
            if display_game:
                print(current_letter + f' makes a move to square {square}')
                game.display_board()
                print('')

            if game.current_winner:
                if display_game:
                    print(current_letter + ' wins!')
                return current_letter

            current_letter = 'O' if current_letter == 'X' else 'X'

    if display_game:
        print('It\'s a tie!')


class RandomPlayer:
    # initializes the player with the letter('x' or 'o')
    def __init__(self, letter): 
        self.letter = letter
    
    # returns a random valid move from available moves on the board
    def get_move(self, game):
        return random.choice(game.available_moves())


class HumanPlayer:
    def __init__(self, letter):
        self.letter = letter

    def get_move(self, game):
        valid_square = False
        val = None
        while not valid_square:
            square = input(self.letter + '\'s turn. Choose move (0-8): ')
            try:
                val = int(square)
                if val not in game.available_moves():
                    raise ValueError
                valid_square = True
            except ValueError:
                print('Invalid square. Try again.')
        return val


class AIPlayer:
    # initializes the player with letter ('x' or 'o')
    def __init__(self, letter):
        self.letter = letter
    
    # uses the maxmin algo to determine the best move 
    def get_move(self, game):
        if len(game.available_moves()) == 9:
            square = random.choice(game.available_moves())
        else:
            square = self.minimax(game, self.letter)['position']
        return square

    # the maxmin algo recursively evaluates possible moves to determine optimal move 
    def minimax(self, state, player):
        max_player = self.letter
        other_player = 'O' if player == 'X' else 'X'

        if state.current_winner == other_player:
            return {'position': None, 'score': 1 * (state.count_empty_squares() + 1) if other_player == max_player else -1 * (state.count_empty_squares() + 1)}
        elif not state.has_empty_squares():
            return {'position': None, 'score': 0}

        if player == max_player:
            best = {'position': None, 'score': -float('inf')}
        else:
            best = {'position': None, 'score': float('inf')}

        for possible_move in state.available_moves():
            state.execute_move(possible_move, player)
            sim_score = self.minimax(state, other_player)

            state.board[possible_move] = ' '
            state.current_winner = None
            sim_score['position'] = possible_move

            if player == max_player:
                if sim_score['score'] > best['score']:
                    best = sim_score
            else:
                if sim_score['score'] < best['score']:
                    best = sim_score
        return best


if __name__ == '__main__':
    x_player = HumanPlayer('X')
    o_player = AIPlayer('O')
    game = TicTacToe()
    play_game(game, x_player, o_player, display_game=True)

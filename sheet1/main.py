import numpy as np
import math

square_dim = 4
discount_factor = 0.8

arr = np.full((square_dim, square_dim), -1)

arr[0][0] = 0
arr[square_dim-1][square_dim-1] = 0


""" states = np.full(square_dim, )
states[0] =  """

#set initial state to bottom left state
current_state = square_dim * (square_dim -1)

terminal_states = [0, square_dim ** 2 - 1]


while current_state not in terminal_states:

    #get coordiantes
    x = math.floor(square_dim / current_state)
    y = square_dim % current_state

    print(x, y)
    break



print(arr)
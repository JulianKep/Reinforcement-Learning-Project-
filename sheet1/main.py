import numpy as np
import math
import random
import time

def state_to_coordinate(current_state, square_dim):
    x = math.floor(square_dim / current_state)
    y = square_dim % current_state
    return(x,y)

square_dim = 4
discount_factor = 0.8

arr = np.full((square_dim, square_dim), -1)

arr[0][0] = 0
arr[square_dim-1][square_dim-1] = 0

print("initial array:\n", arr)

#set initial state to bottom left state
current_state = square_dim * (square_dim -1)

print("initial state:", current_state)


terminal_states = [0, square_dim ** 2 - 1]



while current_state not in terminal_states:

    x = current_state // square_dim
    y = current_state % square_dim

    direction = random.randint(0, 3)

    match direction:
        case 0:  # up
            if x - 1 >= 0:
                x -= 1

        case 1:  # right
            if y + 1 <= square_dim -1:
                y += 1

        case 2:  # down
            if x + 1 <= square_dim -1:
                x += 1

        case 3:  # left
            if y - 1 >= 0:
                y -= 1

    current_state = x * square_dim + y

    time.sleep(0.5)
    print(x, y, current_state)


print(arr)
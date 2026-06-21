# Commands
pip freeze > requirements.txt
pip install -r requirements.txt

source env/bin/activate


# Standard Rules of the Game of "cheat"

- a complete set of 32 standard playing cards a distributed evenly among n players
- the goal is to get rid of all cards on ones hand, the game is played clockwise, player after player
- the player with the 7 of clubs begins, places a card face down and names its value (7, 8, king, etc.)
- the next player needs to place down a card of the same value, face down on the stack
- any player can decide, if its their turn, to doubt that the previous player actually placed the card that was
called for by the first person. they can choose to look at the card, if its the one that was called for, the
revealing player has to take the entire stack into their hand, if not, the player that placed the false card
has to do so instead.
- jacks may never be openly placed, they have to be concealed as other cards
- if at any point a player has all four jacks, that player looses the game
- the player that first gets rid of all cards, wins the game.





# Environment Considerations for the game of cheat

## State Space

To make a strategic decision in the Game of Cheat, multiple things need to be considered.
The State consists of:

- The Cards that are in the agents hand
- the card that is currently called
- the amount of cards on the stack

potentially:
- the amount of cards other players have on hand
- 

## Action Space

- place a specific card from the hand (including jack)
- call a bluff
- bluff

## How do we measure Success 

- Game is won (number of cards is 0)
- number of cards is decreased by one








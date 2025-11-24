This is my flappy bird AI project from my machine learning course. It was my final project, and I decided to go a different route and make it a personal project.
I built a neuroevolution-based Flappy Bird AI in Python with pygame. I found a random github repo of somebody's pygame flappy bird (I unfortunately lost the link so sorry guy)
and used a genetic algorithm to evolve weights of a neural network controller that plays the flappy bird game.

Tech used:
- Python, Pygame, Matplotlib (to map fitness)

- Basic GA evolving a small neural network
- 5 inputs with 1 hidden layery containing 24 nodes
- used np.tanh() >.2 for final output (flap / not flap)

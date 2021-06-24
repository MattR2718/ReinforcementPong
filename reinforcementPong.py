import numpy as np  #for array and random
from PIL import Image  #for creating visual of env
import cv2  #for showing visual live
import matplotlib.pyplot as plt  #for graphing mean rewards over time
import pickle  #to save/load Q-Tables
from matplotlib import style  #to make charts
import time  #keep track of saved Q-Tables.

style.use("ggplot")  # setting our style!

#20 x 20 Grid
SIZE = 20
EPISODES = 1000
MOVE_PENALTY = 1
MISS_PENALTY = 100
HIT_REWARD = 5
#STEPS = 200

epsilon = 0.9
EPS_DECAY = 0.9998

SHOW_EVERY = 1000

start_q_table = None # or filename

LEARNING_RATE = 0.1
DISCOUNT = 0.95

COLOUR = (255, 255, 0)

hit = False

class paddle:
	def __init__(self, side='r'):
		if side == 'r':
			self.x = SIZE - 2
		elif side == 'l':
			self.x = 1
		else:
			print("ERROR PLAYER SIDE INITIALISATION INVALID")
		self.y = [int((SIZE / 2) - 2), int((SIZE / 2) - 1), int((SIZE / 2)), int((SIZE / 2) + 1)]
	
	def action(self, choice):
		if choice == 0:
			self.move(-1)
		elif choice == 1:
			self.move(0)
		elif choice == 2:
			self.move(1)	
	
	def move(self, y=False):
		#print("MOVE")
		if not y:
			#print("NOT Y")
			add = np.random.randint(-1, 2)
			for num in range(4):
				self.y[num] += add
		else:
			#print("ELSE")
			for num in range(4):
				self.y[num] += y
		#print(self.y)

		
		if self.y[0] < 0:
			for i in range(4):
				self.y[i] = i
		elif self.y[3] > SIZE - 1:
			for i in range(4):
				self.y[i] = SIZE - (4 - i)

class circle:
	def __init__(self):
		self.x = int(SIZE / 2)
		self.y = int(SIZE / 2)
		self.v = np.array([2, 0])

	def move(self, player):
		if not ball.collide_player(player) and not ball.line_collide_player(player):
			self.x += self.v[0]
			self.y += self.v[1]
			
		if self.y <= 0:
			self.y = 0
			self.v[1] *= -1
		elif self.y >= SIZE - 1:
			self.y = SIZE - 1
			self.v[1] *= -1

		if self.x <= 0:
			self.x = 0
			self.v[0] *= -1
			#self.v[1] *= -1
		elif self.x > 18:
			self.x = 19

	def collide_player(self, player):
		#self.x += self.v[0]
		#self.y += self.v[1]
		if self.x == player.x - 1 or self.x == player.x:
			#print("self.x == player.x - 1")
			for i in range(4):
				if self.y == player.y[i]:
					#print("self.y == player.y[i]")
					if i == 1 or i == 2:
						#print("i = 1 or 2")
						if self.v[1] >= 0:
							#print("self.v[1] = 0.5")
							self.v = [self.v[0], 1]
							#print(self.v[1])
						else:
							self.v = [self.v[0], -1]
					elif i == 0 or i == 3:
						if self.v[1] > 0:
							self.v = [self.v[0], 2]
						else:
							self.v = [self.v[0], -2]
					self.v[0] *= -1
					
					self.x += self.v[0]
					self.y += self.v[1]

					#print("TRUE")
					return True
			#print("FALSE")			
			return False
		else:
			#print("FALSE")
			return False

	def line_collide_player(self, player):
		pos1 = (self.x, self.y)
		pos2 = ((self.x + self.v[0]), (self.y + self.v[1]))
		
		#m = (y1 - y2)/(x1 - x2)
		m = (pos1[1] - pos2[1]) / (pos1[0] - pos2[0])
		#c - (x1 * y2 - x2 * y1) / (x1 - x2)
		c = ((pos1[0] * pos2[1]) - (pos2[0] * pos1[1])) / (pos1[0] - pos2[0])

		y = (m * player.x) + c
		
		if pos2[0] > 18 and y >= player.y[0] and y < player.y[3] + 1:
			#self.x = player.x - 1
			#self.y = int(y)
			#self.v[0] *= -1
			if self.y == player.y[1] or self.y == player.y[2]:
				if self.v[1] > 0:
					self.v = [self.v[0], 1]
				else:
					self.v = [self.v[0], -1]
			elif self.y == player.y[0] or self.y == player.y[3]:
				if self.v[1] > 0:
					self.v = [self.v[0], 2]
				else:
					self.v = [self.v[0], -2]
			hit = True

			self.x += self.v[0]
			self.y += self.v[1]

			#print("LINE TRUE")
			return True
		else:
			#print("LINE FALSE")
			return False

#If have q table then load
#Otherwise make q_table and fill with random numbers
if start_q_table == None:
	#q_table of all possible states * number of actions
	q_table = np.random.uniform(low=-5, high=0, size=(SIZE, SIZE, 3))
	#np.savetxt('qtable.txt', q_table)
else:
	with open(start_q_table, "rb") as f:
		q_table = pickle.load(f)

episode_rewards = []
for episode in range(EPISODES):
	#print("CALLING PLAYER")
	player = paddle()
	#print("CALLING BALL")
	ball = circle()
	
	if episode % SHOW_EVERY == 0:
		print(f"ON EPISODE: {episode}, EPSILON: {epsilon}")
		print(f"{SHOW_EVERY} EPISODE MEAN: {np.mean(episode_rewards[-SHOW_EVERY:])}")
		show = True
	else:
		show = False
	
	episode_reward = 0
	#Run game until miss ball
	while ball.x < SIZE - 1:

		#time.sleep(1)

		#print(ball.x , ball.y, ball.v)

		#print("GET ACTION")
		obs = (ball.x, ball.y)
		if np.random.random() > epsilon:
			# Get action from Q table
			action = np.argmax(q_table[obs])
		else:
			# Get random action
			action = np.random.randint(0, 3)
		#print(action)

		#print("PLAYER ACTION")
		player.action(action)
		#print("BALL MOVE")
		ball.move(player)

		#print("REWARDS")
		if hit:
			reward = HIT_REWARD
			hit = False
		elif ball.x > SIZE - 1:
			reward = -MISS_PENALTY
		else:
			reward = -MOVE_PENALTY
		
		#print("CALCULATE NEW_Q")
		new_obs = (ball.x, ball.y)
		max_future_q = np.max(q_table[new_obs])
		current_q = q_table[obs][action]

		if reward == HIT_REWARD:
			new_q = HIT_REWARD
		elif reward == -MISS_PENALTY:
			new_q = -MISS_PENALTY
		else:
			new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

		#print("UPDATE Q_TABLE")
		q_table[obs][action] = new_q

		if show:
			#SIZE x SIZE grid of 3 values (BGR)
			env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
			#Set colours
			env[ball.y][ball.x] = COLOUR
			for i in range(4):
				env[player.y[i]][player.x] = COLOUR
			
			img = Image.fromarray(env, "RGB")
			img = img.resize((300, 300))
			cv2.imshow("Environment", np.array(img))
			
			#If we get the food or hit an enemy
			#Wait 500ms and for "q" to be pressed
			if reward == HIT_REWARD or reward == -MISS_PENALTY:
				if cv2.waitKey(500) & 0xFF == ord("q"):
					break
			else:
				if cv2.waitKey(1) & 0xFF == ord("q"):
					break

		#print("----------------------------------")

		episode_reward += reward
		
	episode_rewards.append(episode_reward)
	epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((int(EPISODES / 10),)) / SHOW_EVERY, mode="valid")

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"reward {SHOW_EVERY}")
plt.xlabel("Episode #")
plt.show()

with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
	pickle.dump(q_table, f)

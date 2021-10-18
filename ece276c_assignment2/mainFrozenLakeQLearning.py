import numpy as np
import time
import random
import gym
from gym import envs
#import universe
import matplotlib.pyplot as plt
import random
random.seed(3)
np.random.seed(3)


class FrozenLakeQLearning():
    def __init__(self):
        self.env = gym.make('FrozenLake-v0')
        self.gamma = 0.99
        self.number_episodes = 5000
        self.learning_rate = 0.1
        self.max_steps = 100
        #self.epsilon =

        return

        # Question 2 Part 1.1
    def evaluate_vary_alpha(self):
        alpha = [0.05, 0.1, 0.25, 0.5]

        for learning_rate in alpha:
            self.learning_rate = learning_rate
            Q_table, iteration_success_rate  = self.Q_learning()
            #Q_table, iteration_success_rate  = frozenlake.Q_learning()
            plt.figure(figsize=(8,6.3))
            plt.plot(iteration_success_rate, linewidth=4, label= 'Learning rate a = %.2f' %self.learning_rate)
            plt.xlabel('Number of episode (x100)', fontsize=18)
            plt.ylabel('Success rate', fontsize=18)
            #plt.legend(fontsize=18)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.grid()
            plt.savefig('Learningrat%.2f.png' %self.learning_rate, transparent = True)
        return


        # Question 2 Part 1.2
    def evaluate_vary_gamma(self):
        gamma= [0.9, 0.95, 0.99]

        for discount in gamma:
            self.gamma = discount
            Q_table, iteration_success_rate  = self.Q_learning()
            #Q_table, iteration_success_rate  = frozenlake.Q_learning()
            plt.figure(figsize=(8,6.3))
            plt.plot(iteration_success_rate, linewidth=4, label= 'Discount factor r = %f' %self.gamma)
            plt.xlabel('Number of episode (x100)', fontsize=18)
            plt.ylabel('Success rate', fontsize=18)
            #plt.legend(fontsize=18)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.grid()
            plt.savefig('Discount%.2f.png' %self.gamma, transparent = True)

        return

        # Question 2 Part 2
    def Q_learning_self_exploration(self):
        #Initialize Q table
        iteration_success_rate = []
        Q_table = np.zeros((self.env.nS, self.env.nA))
        for episode in range(self.number_episodes):
            s = self.env.reset()
            #step = 0
            done = False
            total_rewards = 0

            for step in range(self.max_steps):
                #epsilon = 1 - episode/self.number_episodes
                epsilon = 1/(1+np.exp(0.001*episode))
                random_number = random.uniform(0, 1)

                if random_number > epsilon:
                    action = np.argmax(Q_table[s, :])

                else:
                    action = self.env.action_space.sample()

                s_next, reward, done, info = self.env.step(action)

                Q_table[s, action] = Q_table[s, action] + self.learning_rate * (reward + self.gamma*np.max(Q_table[s_next, :]) - Q_table[s, action])


                s = s_next

                if(done):
                    break

            if(episode % 100 == 0):
                iteration_success_rate.append(self.EvaluateQvalue(Q_table, 100))

        return Q_table, iteration_success_rate




    def EvaluateQvalue(self, Q, num_times):

        num_success = 0

        for i in range(num_times):
            s0 = self.env.reset()
            reach_goal = 0
            num_steps = 0
            st = s0
            while True:
                action = np.argmax(Q[st,:])
                s_next, reward, is_terminal, debug_info = self.env.step(int(action))

                st = s_next

                reach_goal += reward
                num_steps += 1

                if is_terminal:
                    break
            num_success += reach_goal

        success_rate = float(num_success/num_times)

        return success_rate



    def Q_learning(self):
        #Initialize Q table
        iteration_success_rate = []
        Q_table = np.zeros((self.env.nS, self.env.nA))
        for episode in range(self.number_episodes):
            s = self.env.reset()
            #step = 0
            done = False
            total_rewards = 0

            for step in range(self.max_steps):
                epsilon = 1 - episode/self.number_episodes
                random_number = random.uniform(0, 1)

                if random_number > epsilon:
                    action = np.argmax(Q_table[s, :])

                else:
                    action = self.env.action_space.sample()

                s_next, reward, done, info = self.env.step(action)

                Q_table[s, action] = Q_table[s, action] + self.learning_rate * (reward + self.gamma*np.max(Q_table[s_next, :]) - Q_table[s, action])


                s = s_next

                if(done):
                    break

            if(episode % 100 == 0):
                iteration_success_rate.append(self.EvaluateQvalue(Q_table, 100))

        return Q_table, iteration_success_rate



if __name__== '__main__':
    frozenlake = FrozenLakeQLearning()

    #frozenlake.evaluate_vary_alpha()
    #frozenlake.evaluate_vary_gamma()
    Q_table, iteration_success_rate  = frozenlake.Q_learning_self_exploration()
    plt.figure(figsize=(8,6.3))
    plt.plot(iteration_success_rate, linewidth=4)
    plt.xlabel('Number of episode (x100)', fontsize=18)
    plt.ylabel('Success rate', fontsize=18)
    #plt.legend(fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid()
    plt.savefig('selfQlearning.png', transparent = True)

    #Q_table, iteration_success_rate  = frozenlake.Q_learning()
    # plt.figure()
    # plt.plot(iteration_success_rate)
    # plt.show()


    # Run Test policy with a specific policy

import numpy as np
import time
import gym
from gym import envs
#import universe
import matplotlib.pyplot as plt


class FrozenLake():
    def __init__(self):
        self.env = gym.make('FrozenLake-v0')
        self.gamma = 1.0
        self.transitions = {s: {a: [] for a in range(self.env.nA)} for s in range(self.env.nS)}
        self.rewards = {s: {a: {s_next: 0 for s_next in range(self.env.nS)} for a in range(self.env.nA)} for s in range(self.env.nS)}
        self.num_data = 10**5
        self.LearnModel()
        return



        # Question 1 Part 3
    def TestPolicy(self, policy, num_times):

        num_success = 0

        for i in range(num_times):
            s0 = self.env.reset()
            reach_goal = 0
            num_steps = 0
            st = s0
            while True:
                s_next, reward, is_terminal, debug_info = self.env.step(int(policy[st]))

                st = s_next

                reach_goal += reward
                num_steps += 1

                if is_terminal:
                    break
            num_success += reach_goal

        success_rate = float(num_success/num_times)

        return success_rate

        # Question 1 Part 4. The learned model is self.transitions, reward is self.rewards
    def LearnModel(self):
        samples = []
        for i in range(self.num_data):
            s0 = self.env.reset()
            reach_goal = 0
            num_steps = 0
            st = s0
            while True:
                action = self.env.action_space.sample()
                s_next, reward, is_terminal, debug_info = self.env.step(action)
                samples.append((s_next, reward, st, action))
                st = s_next

                if is_terminal:
                    break

        frequency = np.zeros((self.env.nS, self.env.nA, self.env.nS))

        frequency_s_a =  np.zeros((self.env.nS, self.env.nA))

        for s_next, reward, st, action in samples:
            frequency[st][action][s_next] += 1
            frequency_s_a[st][action] +=1
            self.rewards[st][action][s_next] += reward


        for st in range(self.env.nS):
            for action in range(self.env.nA):
                for s_next in range(self.env.nS):
                    if frequency[st][action][s_next] == 0: continue
                    self.rewards[st][action][s_next] = self.rewards[st][action][s_next]/frequency[st][action][s_next]

        for st in range(self.env.nS):
            for action in range(self.env.nA):
                for s_next in range(self.env.nS):
                    if frequency[st][action][s_next] == 0: continue
                    self.transitions[st][action].append((frequency[st][action][s_next]/frequency_s_a[st][action], s_next, self.rewards[st][action][s_next]))


        return





    def PolicyFromValue(self, value):
        policy = np.zeros(self.env.nS)

        for s in range(self.env.nS):
            Q = np.zeros(self.env.nA)

            for a in range(self.env.nA):
                Q[a] = sum([p* (r + self.gamma *value[s_next]) for p, s_next, r in self.transitions[s][a]])

                policy[s] = np.argmax(Q)
                #print(Q)

        return policy

        # Question 1 Part 5
    def PolicyEval(self, policy):
        value = np.zeros(self.env.nS)
        eps = 0.01
        while True:
            p_value = np.copy(value)
            for s in range(self.env.nS):
                a = policy[s]
                value[s] = sum([p * (r + self.gamma * p_value[s_next]) for p, s_next, r in self.transitions[s][a]])

            if (np.sum((np.fabs(p_value - value))) <= eps):
                print('Policy Evaluation Done')
                break


        return value

        # Question 1 Part 5
    def PolicyIter(self, maxIteration):
        policy = np.random.choice(self.env.nA, size=(self.env.nS))
        iteration_success_rate = []

        for i in range(maxIteration):
            v_policy = self.PolicyEval(policy)
            policy_updated = self.PolicyFromValue(v_policy)
            #if (np.all(policy == policy_updated)):
                #break

            policy = policy_updated
            #print(v_policy)
            iteration_success_rate.append(self.TestPolicy(policy, 100))
            #print(iteration_success_rate)
        return policy, iteration_success_rate

        # Question 1 Part 6
    def ValueIter(self, maxIteration):
        value = np.zeros(self.env.nS)
        eps = 0.001
        iteration_success_rate = []

        for i in range(maxIteration):
            p_value = np.copy(value)
            for s in range(self.env.nS):
                Q = [sum([p * (r + p_value[s_next]) for p, s_next, r in self.transitions[s][a]]) for a in range(self.env.nA)]
                value[s] = max(Q)

            policy = self.PolicyFromValue(value)

            iteration_success_rate.append(self.TestPolicy(policy, 100))

        return policy, iteration_success_rate







if __name__== '__main__':
    frozenlake = FrozenLake()

    #policy = lambda s: (s+1) %4
    policy = dict()
    for i in range(frozenlake.env.nS):
        policy[i] = (i+1) %4

    success_rate = frozenlake.TestPolicy(policy, 100)
    print('Success rate is', success_rate)
    # num_success = 0
    # for i in range(100):
    #     reach_goal, num_steps = frozenlake.TestPolicy(policy)
    #     num_success += reach_goal

    #print(frozenlake.LearnModel(1,2,2))


    #print('Success rate is', frozenlake.TestPolicy(policy, 100))

    # updated_policy1, iteration_success_rate1  = frozenlake.PolicyIter(50)
    # plt.figure(figsize=(8,6.3))
    # plt.plot(iteration_success_rate1, linewidth=4)
    # plt.xlabel('Number of iteration', fontsize=18)
    # plt.ylabel('Success rate', fontsize=18)
    # #plt.legend(fontsize=18)
    # plt.xticks(fontsize=18)
    # plt.yticks(fontsize=18)
    # plt.grid()
    # plt.savefig('policy_iteration.png', transparent = True)
    # #
    # # plt.figure()
    # # plt.plot(iteration_success_rate)
    # # plt.show()
    #
    #
    # updated_policy2, iteration_success_rate2  = frozenlake.ValueIter(50)
    #
    #
    # plt.figure(figsize=(8,6.3))
    # plt.plot(iteration_success_rate2, linewidth=4)
    # plt.xlabel('Number of iteration', fontsize=18)
    # plt.ylabel('Success rate', fontsize=18)
    # #plt.legend(fontsize=18)
    # plt.xticks(fontsize=18)
    # plt.yticks(fontsize=18)
    # plt.grid()
    # plt.savefig('value_iteration.png', transparent = True)

    # Run Test policy with a specific policy

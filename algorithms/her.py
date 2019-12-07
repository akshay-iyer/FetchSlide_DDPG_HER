import numpy as np

class HER:
    def __init__(self, reward_fn):
        # probability to include an HER replay between normal replays
        #self.future_p = 1 - (1. / (1 + replay_k))
        self.reward_fn = reward_fn

    def sample_goals_her (self, buffer_temp, num_transitions):
        T = buffer_temp['actions'].shape[0]

        batch_size = num_transitions
        # select which rollouts and which timesteps to be used

        t_samples = np.random.randint(T-1, size=batch_size)
        transitions = {key: buffer_temp[key][t_samples].copy() for key in buffer_temp.keys()}
        # her idx
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        # replace go with achieved goal
        future_ag = buffer_temp['ag'][future_t]
        transitions['g'][her_indexes] = future_ag
        # to get the params to re-compute reward
        transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], None), 1)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

    def _apply_hindsight(self, buffer_temp):
        #print("buffer_temp:")
        #print(buffer_temp)
        num_transitions = len(buffer_temp['actions'])
        new_desired_goal = buffer_temp['ag_next'][-1]
        hind_experiences = buffer_temp.copy()
        hind_experiences['r'] = []
        for i in range(num_transitions):
            hind_experiences['g'][i] = new_desired_goal
            reward = self.reward_fn(hind_experiences['ag_next'][i], hind_experiences['g'][i], None)
            hind_experiences['r'].append(reward)
        hind_experiences['g_next'] = hind_experiences['g'][1:,:]

        return hind_experiences
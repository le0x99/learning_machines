class Agent(robobo.SimulationRobobo):
    def __init__(self, ip='192.168.1.133',
                 port=19997,
                 act_range = (20,100),
                 min_act = False,
                 act_granularity = 100,
                 discount=.95
                 ):
        super().__init__()
        super().connect(address=ip, port=port)
        self.discount = discount
        self.states = []
        self.actions = []
        self.training_results = {"critic" : [], "actor" : [], "reward" : []}
        self.min_act = act_range[0]
        self.max_act = act_range[1]
        random_policy = lambda state : np.random.randint(self.min_act,
                                                         self.max_act, 2)
        self.critic = Critic()
        self.actor = Actor()
        self.optim_c = optim.Adam(self.critic.parameters(), lr=0.0001)
        self.optim_a = optim.Adam(self.actor.parameters(), lr=0.0001)
        self.fps = 1000 / act_granularity
        self.act_granularity = act_granularity
        
    def start_sim(self): return self.play_simulation()
    def end_sim(self): return self.stop_world()
    def pause_sim(self): return self.pause_simulation()

    def mmove(self, r, l, t, cd=2.5):
        time.sleep(cd)
        self.move(r, l, t)
    def run_episode(self, n_steps=1000):
        self.n_steps = n_steps
        self.cumul_reward = 0.
        if self.is_simulation_running():
            self.end_sim()
        self.episode()
    def episode(self):
        self.cumul_reward = 0.
        print("Initializing simulation")
        self.start_sim()
        print("Starting Episode")
        print(f"FPS = {self.fps}")
        for step in range(self.n_steps):
            self.act()
        print("Closing simulation")
        self.end_sim()
    def get_state(self, use_cam = False,
                  normalize_ir = True,
                 invert_ir = True):
        # to do : edit parent + normalize
        X1 = torch.tensor([[ .2 if not _ else _ for _ in self.read_irs() ]])
        X1 = (X1 - 0.) / (.2 - 0.) if normalize_ir else X1
        X1 = 1 - X1 if invert_ir else X1
        X1 = X1.to(torch.float32)
        # X2 = self.get_image_front()
        if not use_cam:
            self.states.append(X1)
            return X1
        else:
            pass

    def act(self):
        state = self.states[-1] if len(self.states) > 0 else self.get_state()
        # actor
        action_dists = self.actor(state)
        action = action_dists.rsample()
        #action = torch.maximum(action, torch.Tensor([self.min_act, self.min_act])) if self.min_act else output
        # actuate
        control = action.clone().detach().numpy()
        self.actions.append(control)
        r, l = control[0][0], control[0][1]
        self.move(r, l, self.act_granularity)
        # observe next state and reward
        next_state = self.get_state()
        reward_next_state = action.sum() / 200 - next_state.sum() * 10
        #reward_next_state = action.abs().sum() - next_state.sum()
        # critic update
        self.optim_c.zero_grad()
        prediction_state = player.critic(state)
        prediction_next_state = player.critic(next_state)
        #target = self.discount * prediction_next_state.detach() + reward_next_state
        target = self.discount * prediction_next_state+ reward_next_state
        target = target.detach()
        loss_critic = ((prediction_state-target)**2).mean()
        loss_critic.backward()
        self.optim_c.step()
        # actor update
        advantage = reward_next_state + self.discount * prediction_next_state - prediction_state
        log_probs = action_dists.log_prob(action)
        loss_actor = (-log_probs * advantage.detach()).mean()
        self.optim_a.zero_grad()
        loss_actor.backward()
        nn.utils.clip_grad_norm_([p for g in self.optim_a.param_groups for p in g["params"]], .5) 
        self.optim_a.step()    
        self.cumul_reward += reward_next_state.item()
        self.training_results["critic"].append(loss_critic.item())
        self.training_results["actor"].append(loss_actor.item())
        self.training_results["reward"].append(reward_next_state.item())
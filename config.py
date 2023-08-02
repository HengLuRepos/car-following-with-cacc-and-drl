class Config:
    def __init__(self, max_acc = 3, tau=0.8, max_v=30):
        self.buffer_size=None
        self.batch_size=256
        self.gamma = 0.9
        self.tau = 0.99
        self.pi_lr=3e-4
        self.v_lr = 3e-4
        self.explore_noise=0.1
        self.target_noise=0.2
        self.noise_clip=0.5
        self.policy_delay=2
        self.seed=0

        self.start_steps = 5000
        self.max_acc = max_acc
        self.tau_t = tau #time
        self.max_v = max_v

        self.max_timestamp = 1000000
        self.a_low = -self.max_acc
        self.a_high = self.max_acc
        self.eval_freq = 5000
        self.eval_epoch = 1
        self.update_freq = 1
        

class Config:
    def __init__(self, max_acc = 3, tau=0.8, max_v=30):
        self.buffer_size=100000
        self.batch_size=256
        self.gamma = 0.9
        self.rho = 0.995
        self.pi_lr=5e-4
        self.v_lr = 5e-4
        self.explore_noise=0.1
        self.seed=0

        self.start_steps = 5000
        self.max_acc = max_acc
        self.tau = tau
        self.max_v = max_v

        self.max_timestamp = 100000
        self.a_low = -self.max_acc
        self.a_high = self.max_acc
        

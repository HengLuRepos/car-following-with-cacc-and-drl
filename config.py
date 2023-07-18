class Config:
    def __init__(self, action_max):
        self.buffer_size=2000
        self.batch_size=64
        self.gamma = 1
        self.rho = 0.995
        self.lr=5e-4
        
        self.layer_size=64
        self.n_layers=3

        self.start_steps = 200
        self.max_action = action_max
class Config:
    def __init__(self, action_max):
        self.buffer_size=100000
        self.batch_size=100
        self.gamma = 0.9
        self.rho = 0.995
        self.lr=5e-4
        
        self.layer_size=128
        self.n_layers=3

        self.start_steps = 5000
        self.max_action = action_max
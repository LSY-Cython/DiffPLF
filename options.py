import torch

class Options:
    def __init__(self, isTrain, isRefine):
        self.isConstrain = True  # validate efficacy of covariates
        self.isCrossAttention = True  # validate efficacy of cross-attention mechanism
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.init_lr = 1e-3
        self.n_epochs = 200
        self.seq_len = 96  # 15-min resolution
        self.input_dim = 1
        self.d_diff = 32
        self.d_cond = 32
        self.d_cross = 32
        self.cond_dim = 7  # N_past + 2
        self.num_heads = 4
        self.beta_start = 1e-4
        self.beta_end = 0.5
        self.n_steps = 300  # diffusion step T, default 200
        self.schedule = "quadratic"
        self.weight_sign = "normal"
        if not self.isConstrain:
            self.cond_dim -= 2  # remove weather vars
            self.weight_sign = "no constrain"
        if not self.isCrossAttention:
            self.weight_sign = "no cross"
        if isTrain:
            self.batch_size = 16
            self.shuffle = True
        elif isRefine:
            self.batch_size = 16
            self.shuffle = True
            self.init_lr = 2e-4
            self.n_epochs = 100
            self.eta = 0.001
        else:
            self.batch_size = 1
            self.shuffle = False
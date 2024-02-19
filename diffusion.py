import torch.utils.data
import numpy as np
import scipy.signal as sig
from network import *
from dataset import process_input
from utils import *
import os
import time

class cDDPM:
    def __init__(self, opt, data_loader):
        super().__init__()
        self.denoiser = Denoiser(opt).to(opt.device)
        self.opt = opt
        self.n_steps = opt.n_steps
        self.beta = torch.linspace(opt.beta_start**0.5, opt.beta_end**0.5, opt.n_steps, device=opt.device)**2
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sigma2 = torch.cat((torch.tensor([self.beta[0]], device=opt.device), self.beta[1:]*(1-self.alpha_bar[0:-1])/(1-self.alpha_bar[1:])))
        self.optimizer = torch.optim.Adam(self.denoiser.parameters(), lr=opt.init_lr, betas=(0.5, 0.999))
        self.data_loader = data_loader
        self.loss_func = nn.MSELoss()
        p1, p2 = int(0.5*opt.n_epochs), int(0.75*opt.n_epochs)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[p1, p2], gamma=0.1)
        self.infer_time = []

    def gather(self, const, t):
        return const.gather(-1, t).view(-1, 1, 1)

    def q_xt_x0(self, x0, t):
        alpha_bar = self.gather(self.alpha_bar, t)
        mean = (alpha_bar**0.5)*x0
        var = 1 - alpha_bar
        return mean, var

    def q_sample(self, x0, t, eps):
        mean, var = self.q_xt_x0(x0, t)
        return mean + (var**0.5)*eps

    def p_sample(self, xt, history, weather, day_type, ev_num, t):
        eps_theta = self.denoiser(xt, history, weather, day_type, ev_num, t)
        alpha_bar = self.gather(self.alpha_bar, t)
        alpha = self.gather(self.alpha, t)
        eps_coef = (1 - alpha)/(1 - alpha_bar)**0.5
        mean = (xt - eps_coef*eps_theta)/(alpha**0.5)
        var = self.gather(self.sigma2, t)
        if (t == 0).all():
            z = torch.zeros(xt.shape, device=xt.device)
        else:
            z = torch.randn(xt.shape, device=xt.device)
        return mean + (var**0.5)*z

    def cal_train_loss(self, x0, history, weather, day_type, ev_num):
        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, eps=noise)
        eps_theta = self.denoiser(xt, history, weather, day_type, ev_num, t)
        return self.loss_func(noise, eps_theta)

    def cal_refine_loss(self, x0, m0, history, weather, day_type, ev_num, pred_len):
        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, eps=noise)
        mt = self.q_sample(m0, t, eps=noise)
        if pred_len != 96:
            mt[:, 0:96-pred_len, :] = xt[:, 0:96-pred_len, :]
        x_eps = self.denoiser(xt, history, weather, day_type, ev_num, t)
        m_eps = self.denoiser(mt, history, weather, day_type, ev_num, t)
        prior_loss = self.loss_func(noise, x_eps)
        reg_loss = self.loss_func(x_eps, m_eps)
        refine_loss = prior_loss + self.opt.eta*reg_loss
        return refine_loss

    def read_data(self, data):
        x0 = data["true"].to(self.opt.device)
        history = data["hist"].to(self.opt.device)
        weather = data["weather"].to(self.opt.device)
        day_type = data["day"].to(self.opt.device)
        ev_num = data["num"].to(self.opt.device)
        return x0, history, weather, day_type, ev_num

    def train(self):
        epoch_loss, epoch_time = [], []
        for epoch in range(self.opt.n_epochs):
            batch_loss = []
            start_time = time.time()
            for i, data in enumerate(self.data_loader):
                x0, history, weather, day_type, ev_num = self.read_data(data)
                self.optimizer.zero_grad()
                loss = self.cal_train_loss(x0, history, weather, day_type, ev_num)
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            end_time = time.time()
            epoch_loss.append(np.mean(batch_loss))
            epoch_time.append(end_time - start_time)
            print(f"epoch={epoch}/{self.opt.n_epochs}, loss={epoch_loss[-1]}, time={epoch_time[-1]}s")
            self.lr_scheduler.step()
            save_path = f"weights/train/{self.opt.weight_sign}/epoch{epoch}.pt"
            torch.save(self.denoiser.state_dict(), save_path)
        print(f"Average training time: {np.mean(epoch_time)}s")
        plot_training_loss(epoch_loss)

    def load_weight(self, weight_path):
        weight = torch.load(weight_path, map_location=self.opt.device)
        self.denoiser.load_state_dict(weight)

    def infer(self, weight_path, n_samples, test_file, isCollect, ev_ratio, pred_len):
        true, history, p_max, weather, day_type, ev_num = process_input(test_file, self.opt.isConstrain)
        sign = test_file.split("/")[-1].rstrip(".pkl")
        with torch.no_grad():
            self.load_weight(weight_path)
            self.denoiser.eval()
            start_time = time.time()
            x = torch.randn([n_samples, self.opt.seq_len, self.opt.input_dim]).to(self.opt.device)
            history = history.unsqueeze(0).repeat(n_samples, 1, 1).to(self.opt.device)
            weather = weather.unsqueeze(0).repeat(n_samples, 1, 1).to(self.opt.device)
            day_type = day_type.unsqueeze(0).repeat(n_samples, 1, 1).to(self.opt.device)
            ev_num = ev_num.unsqueeze(0).repeat(n_samples, 1, 1).to(self.opt.device)
            pred_outcomes = []
            for j in range(0, self.n_steps, 1):
                t = torch.ones(n_samples, dtype=torch.long).to(self.opt.device)*(self.n_steps-j-1)
                x = self.p_sample(x, history, weather, day_type, ev_num*ev_ratio, t)
            if self.opt.isConstrain:  # scaling with EV number
                p_max *= ev_num.squeeze().detach().cpu().numpy().tolist()[0]
            x = x.detach().cpu().numpy().squeeze(2)  # (n_samples, L)
            end_time = time.time()
            self.infer_time.append(end_time-start_time)
            print(f"Inference time on {test_file}: {self.infer_time[-1]}s")
            for i in range(n_samples):
                # x_filt = sig.medfilt(x[i], 5)*p_max
                x_filt = x[i]*p_max*ev_ratio  # just takes effect are generated samples excluding ground truth
                pred_outcomes.append(x_filt.tolist())
            groud_truth = (true*p_max).squeeze().detach().numpy()
            plot_prediction(pred_outcomes, groud_truth, sign, isCollect, pred_len)

    def fine_tuning(self, weight_path, pred_len):
        self.load_weight(weight_path)
        # freeze partial model
        for name, parameter in self.denoiser.named_parameters():
            if "load_encoder" in name or "cond_encoder" in name or "cross_att" in name:
                parameter.requires_grad = False
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.denoiser.parameters()),
                                          lr=self.opt.init_lr, betas=(0.5, 0.999))
        epoch_loss, epoch_time = [], []
        for epoch in range(self.opt.n_epochs):
            batch_loss = []
            start_time = time.time()
            for i, data in enumerate(self.data_loader):
                x0, history, weather, day_type, ev_num = self.read_data(data)
                m0 = data["median"].to(self.opt.device)
                self.optimizer.zero_grad()
                loss = self.cal_refine_loss(x0, m0, history, weather, day_type, ev_num, pred_len)
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            end_time = time.time()
            epoch_loss.append(np.mean(batch_loss))
            epoch_time.append(end_time - start_time)
            print(f"epoch={epoch}/{self.opt.n_epochs}, loss={epoch_loss[-1]}, time={epoch_time[-1]}s")
            self.lr_scheduler.step()
            save_path = f"weights/refine/epoch{epoch}.pt"
            torch.save(self.denoiser.state_dict(), save_path)
        print(f"Average training time: {np.mean(epoch_time)}s")
        plot_refine_loss(epoch_loss)
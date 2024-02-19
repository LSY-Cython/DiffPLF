import torch.nn as nn
import torch.utils.data
from network import Attention
from utils import *
from dataset import *

class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
        self.rnn = nn.LSTM(5, 32, num_layers=1, batch_first=True)
        self.att = Attention(d_model=32,
                             num_heads=4,
                             hq=32,
                             hv=32,
                             hk=32)
        self.out_proj_layer = nn.Sequential(
            nn.Linear(32, 32, bias=True),
            nn.ReLU(),
            nn.Linear(32, 1, bias=True),
            nn.ReLU()
        )

    def forward(self, history):
        hid_enc, (_, _) = self.rnn(history)  # (B, L, d_cond)
        att_enc = self.att(hid_enc, hid_enc, hid_enc)  # (B, L, d_cond)
        output = self.out_proj_layer(att_enc)  # (B, L, 1)
        return output

class QTRG():  # quantile regression
    def __init__(self, data_loader):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_epochs = 200
        self.predictor = Predictor().to(self.device)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=1e-3, betas=(0.9, 0.999))
        self.data_loader = data_loader
        self.loss_func = nn.MarginRankingLoss(reduction='sum')

    def cal_train_loss(self, history, y, q):
        # (B, L)
        y_bar = self.predictor(history).squeeze(2)
        y = y.squeeze(2)
        target = torch.ones_like(y)*q
        over_bias = self.loss_func(y_bar, y, target)
        under_bias = self.loss_func(y, y_bar, 1-target)
        pinball_loss = (over_bias + under_bias) / (y.shape[0]*y.shape[1])
        return pinball_loss

    def train(self, q):
        epoch_loss = []
        for epoch in range(self.n_epochs):
            batch_loss = []
            for i, data in enumerate(self.data_loader):
                y = data["true"].to(self.device)
                history = data["hist"].to(self.device)
                self.optimizer.zero_grad()
                loss = self.cal_train_loss(history, y, q)
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(np.mean(batch_loss))
            print(f"epoch={epoch}/{self.n_epochs}, loss={epoch_loss[-1]}")
            save_path = f"weights/regression/{q}/epoch{epoch}.pt"
            torch.save(self.predictor.state_dict(), save_path)
        plot_training_loss(epoch_loss)

    def infer(self, test_file):
        true, history, p_max, _, _, _ = process_input(test_file, isConstrain=False)
        y_true = true.squeeze().detach().numpy()*p_max
        output_data = {"true": y_true}
        for q in [0.05, 0.25, 0.5, 0.75, 0.95]:
            weight_path = f"weights/regression/{q}.pt"
            weight = torch.load(weight_path, map_location=self.device)
            self.predictor.load_state_dict(weight)
            y_pred = self.predictor(history.unsqueeze(0)).squeeze().detach().numpy()*p_max
            output_data[q] = y_pred
        sign = test_file.split("/")[-1].rstrip(".pkl")
        pkl_path = f"generation/regression/{sign}.pkl"
        with open(pkl_path, "wb") as f:
            pkl.dump(output_data, f)
        img_path = f"generation/regression/{sign}.png"
        plot_regression(y_true, output_data[0.5], output_data[0.25], output_data[0.75], output_data[0.05], output_data[0.95], img_path)

if __name__ == "__main__":
    # 0.05:16, 0.75:48
    data_file = f"dataset/pred_dataset.json"
    data_loader = creat_dataloader(data_file, batch_size=32, shuffle=True, isRefine=False, isConstrain=False)
    model = QTRG(data_loader)
    quantile = 0.5
    # model.train(quantile)

    with open("dataset/pred_dataset.json", "r") as f:
        test_paths = json.load(f)["test"]
    for test_file in test_paths:
        model.infer(test_file)
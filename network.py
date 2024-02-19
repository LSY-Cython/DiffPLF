from torch import nn
import torch
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, d_model, num_heads, hq, hv, hk):
        super(Attention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scaling = float(self.head_dim)**-0.5
        self.q_proj_layer = nn.Linear(hq, d_model, bias=True)
        self.k_proj_layer = nn.Linear(hk, d_model, bias=True)
        self.v_proj_layer = nn.Linear(hv, d_model, bias=True)
        self.norm = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model, bias=True),
            nn.ReLU(),
            nn.Linear(d_model, d_model, bias=True)
        )

    """
    1) (B, L, hq), (B, L, hk), (B, L, hv)
    2) key == value, hk == hv
    3) dq == dk == dv == head_dim
    """
    def forward(self, query, key, value):
        # (L, B, d_model)
        Q = self.q_proj_layer(query).transpose(0, 1)
        K = self.k_proj_layer(key).transpose(0, 1)
        V = self.v_proj_layer(value).transpose(0, 1)
        # (B*num_heads, L, head_dim)
        seq_len, batch_size = Q.shape[0], Q.shape[1]
        Q = Q.reshape(seq_len, batch_size*self.num_heads, self.head_dim).transpose(0, 1)
        K = K.reshape(seq_len, batch_size*self.num_heads, self.head_dim).transpose(0, 1)
        V = V.reshape(seq_len, batch_size*self.num_heads, self.head_dim).transpose(0, 1)
        # (B*num_heads, L, L)
        att_weights = torch.bmm(Q, K.transpose(1, 2))*self.scaling
        att_weights = F.softmax(att_weights, dim=-1)
        # (B*num_heads, L, head_dim)
        att_output0 = torch.bmm(att_weights, V)
        # (B, L, d_model)
        att_output0 = att_output0.transpose(0, 1).reshape(seq_len, batch_size, self.d_model).transpose(0, 1)
        # Add & Norm1
        attn_output1 = att_output0 + key
        attn_output1 = self.norm(attn_output1)
        # Feed forward layer
        attn_output2 = self.feed_forward(attn_output1)
        # Add & Norm2
        attn_output3 = attn_output2 + attn_output1
        attn_output3 = self.norm(attn_output3)
        return attn_output3

class LoadEncoder(nn.Module):
    def __init__(self, opt):
        super(LoadEncoder, self).__init__()
        self.opt = opt
        self.rnn = nn.LSTM(opt.input_dim, opt.d_diff, num_layers=1, batch_first=True)
        self.att = Attention(d_model=opt.d_diff,
                             num_heads=opt.num_heads,
                             hq=opt.d_diff,
                             hv=opt.d_diff,
                             hk=opt.d_diff)
        self.embed = nn.Sequential(
            nn.Linear(opt.d_diff, opt.d_diff),
            nn.ReLU()
        )
        self.out_proj_layer = nn.Sequential(
            nn.Linear(opt.d_diff, opt.d_diff, bias=True),
            nn.ReLU()
        )

    def time_embedding(self, t, hidden_dim, seq_len):  # (B, )
        t = t.view(-1, 1)
        te = torch.zeros(t.shape[0], hidden_dim)
        div_term = 1/torch.pow(10000.0, torch.arange(0, hidden_dim, 2, dtype=torch.float32)/hidden_dim).to(self.opt.device)
        te[:, 0::2] = torch.sin(t*div_term)
        te[:, 1::2] = torch.cos(t*div_term)
        te = te.view(te.shape[0], 1, hidden_dim).repeat(1, seq_len, 1).to(self.opt.device)
        return te

    def forward(self, x, t):
        hid_enc, (_, _) = self.rnn(x)  # (B, L, d_diff)
        time_emb = self.time_embedding(t, self.opt.d_diff, self.opt.seq_len)  # (B, L, d_diff)
        time_emb = self.embed(time_emb)
        hid_enc = hid_enc + time_emb  # (B, L, d_diff)
        att_enc = self.att(hid_enc, hid_enc, hid_enc)  # (B, L, d_diff)
        # x_enc = self.out_proj_layer(hid_enc)
        return att_enc

class CondEncoder(nn.Module):
    def __init__(self, opt):
        super(CondEncoder, self).__init__()
        self.opt = opt
        self.rnn = nn.LSTM(opt.cond_dim, opt.d_cond, num_layers=1, batch_first=True)
        self.att = Attention(d_model=opt.d_cond,
                             num_heads=opt.num_heads,
                             hq=opt.d_cond,
                             hv=opt.d_cond,
                             hk=opt.d_cond)
        self.embed = nn.Sequential(
            nn.Linear(7+1, opt.d_cond),
            nn.ReLU()
        )
        self.out_proj_layer = nn.Sequential(
            nn.Linear(opt.d_cond, opt.d_cond, bias=True),
            nn.ReLU()
        )

    def forward(self, history, weather, day_type, ev_num):
        if self.opt.isConstrain:
            input = torch.cat((history, weather), -1)  # (B, L, N+2)
            disc_emb = self.embed(torch.cat((day_type, ev_num), -1))  # (B, d_cond)
            disc_emb = disc_emb.view(input.shape[0], 1, self.opt.d_cond).repeat(1, self.opt.seq_len, 1)  # (B, L, d_cond)
            hid_enc, (_, _) = self.rnn(input)  # (B, L, d_cond)
            hid_enc = hid_enc + disc_emb  # (B, L, d_cond)
            att_enc = self.att(hid_enc, hid_enc, hid_enc)  # (B, L, d_cond)
            # c_enc = self.out_proj_layer(hid_enc)
        else:
            hid_enc, (_, _) = self.rnn(history)  # (B, L, d_cond)
            att_enc = self.att(hid_enc, hid_enc, hid_enc)  # (B, L, d_cond)
        return att_enc

class CrossAtt(nn.Module):
    def __init__(self, opt):
        super(CrossAtt, self).__init__()
        self.opt = opt
        self.att = Attention(d_model=opt.d_cross,
                             num_heads=opt.num_heads,
                             hq=opt.d_cond,
                             hv=opt.d_diff,
                             hk=opt.d_diff)

    def forward(self, x_emb, c_emb):
        att_enc = self.att(c_emb, x_emb, x_emb)  # (B, L, d_cross)
        return att_enc

class Denoiser(nn.Module):
    def __init__(self, opt):
        super(Denoiser, self).__init__()
        self.load_encoder = LoadEncoder(opt)
        self.cond_encoder = CondEncoder(opt)
        self.cross_att = CrossAtt(opt)
        self.self_att = Attention(d_model=opt.d_cross,
                                  num_heads=opt.num_heads,
                                  hq=opt.d_cross,
                                  hv=opt.d_cross,
                                  hk=opt.d_cross)
        self.out_proj_layer = nn.Sequential(
            nn.Linear(opt.d_cross, opt.d_cross, bias=True),
            nn.ReLU(),
            nn.Linear(opt.d_cross, opt.input_dim, bias=True),  # no activation for Gaussian noise
        )
        self.opt = opt

    # def cal_correlation(self):
    #     ex = torch.mean(self.x_enc, dim=2)  # (B, L)
    #     ec = torch.mean(self.c_enc, dim=2)  # (B, L)
    #     ex_mean = torch.mean(ex, dim=0)
    #     ec_mean = torch.mean(ec, dim=0)
    #     cov_mat = torch.matmul((ex-ex_mean).T, ec-ec_mean)/ex.shape[0]  # (L, L)
    #     cov_loss = torch.mean(cov_mat)
    #     return cov_loss

    def forward(self, x, history, weather, day_type, ev_num, t):
        # input embedding before attention
        x_enc = self.load_encoder(x, t)  # (B, L, d_diff)
        c_enc = self.cond_encoder(history, weather, day_type, ev_num)  # (B, L, d_cond)
        if self.opt.isCrossAttention:
            cross_enc = self.cross_att(x_enc, c_enc)  # (B, L, d_cross)
        else:
            cross_enc = x_enc + c_enc  # element-wise addition
        # cross_enc = cross_enc + self.x_enc  # skip connection
        att_enc = self.self_att(cross_enc, cross_enc, cross_enc)
        output = self.out_proj_layer(att_enc)  # (B, L, input_dim)
        return output
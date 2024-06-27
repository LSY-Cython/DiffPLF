import matplotlib.pyplot as plt
import pickle as pkl
import os
import numpy as np

def plot_training_loss(epoch_loss):
    plt.plot(epoch_loss, label=f"Training loss")
    plt.legend()
    plt.savefig(f"training_loss.png")
    plt.clf()

def plot_refine_loss(epoch_loss):
    plt.plot(epoch_loss, label=f"Fine-tuning loss")
    plt.legend()
    plt.savefig(f"refining_loss.png")
    plt.clf()

def plot_prediction(preds, true, sign, isCollect, pred_len):
    true = np.array(true)[96-pred_len:]
    gen_samples = np.array(preds)[:, 96-pred_len:]
    median = np.percentile(gen_samples, q=50, axis=0)
    if not isCollect:
        img_path = f"generation/PI_{sign}.png"
        plt.plot(true, color="blue", label="Ground truth")
        lb_50 = np.percentile(gen_samples, q=25, axis=0)
        ub_50 = np.percentile(gen_samples, q=75, axis=0)
        lb_90 = np.percentile(gen_samples, q=5, axis=0)
        ub_90 = np.percentile(gen_samples, q=95, axis=0)
        colors = ["lightgreen", "limegreen", "darkgreen"]
        plt.stackplot(np.arange(len(true)), lb_90, lb_50-lb_90, ub_50-lb_50, ub_90-ub_50,
                      colors=["white", colors[0], colors[1], colors[0]], baseline="zero")
        plt.plot(median, color=colors[2], label="Point forecast")
        plt.legend()
        plt.savefig(img_path)
        plt.clf()
        output_data = {"true": true, "median": median, "generation": gen_samples}
        pkl_path = f"generation/{sign}.pkl"
        with open(pkl_path, "wb") as f:
            pkl.dump(output_data, f)
    else:
        pkl_path = f"dataset/median/{sign}.pkl"
        with open(pkl_path, "wb") as f:
            pkl.dump(median.reshape(-1, 1), f)  # (L, 1)
    print(f"{pkl_path} predicted done!")

def plot_regression(true, median, lb_50, ub_50, lb_90, ub_90, img_path):
    colors = ["lightgreen", "limegreen", "darkgreen"]
    plt.stackplot(np.arange(len(true)), lb_90, lb_50 - lb_90, ub_50 - lb_50, ub_90 - ub_50,
                  colors=["white", colors[0], colors[1], colors[0]], baseline="zero")
    plt.plot(median, color=colors[2], label="Point forecast")
    plt.plot(true, color="blue", label="Ground truth")
    plt.legend()
    plt.savefig(img_path)
    plt.clf()
    print(f"{img_path} regressed done!")

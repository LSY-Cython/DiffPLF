import matplotlib.pyplot as plt
import pickle as pkl
import random
import numpy as np

plt.rcParams['font.sans-serif'] = "Times New Roman"

def config_time_axis(interval, y_max):
    plt.xlim((0, 24))
    positions = list(range(0, 96+1, 4*interval))
    ticks = list(range(0, 24+1, 1*interval))
    plt.xticks(positions, ticks, fontsize=15)
    plt.ylim((0, y_max))
    plt.yticks(fontsize=15)

def show_overall_pi(input_folder, days):
    fig = plt.figure(figsize=(12, 15), dpi=200)
    for m in range(1, 12+1, 1):
        day = f"2019-{m}-{days[m-1]}"
        pkl_path = f"{input_folder}/{day}.pkl"
        with open(pkl_path, "rb") as f:
            input_data = pkl.load(f)
        ground_truth = input_data["true"]
        gens = input_data["generation"]
        lb_50 = np.percentile(gens, q=25, axis=0)
        ub_50 = np.percentile(gens, q=75, axis=0)
        lb_90 = np.percentile(gens, q=5, axis=0)
        ub_90 = np.percentile(gens, q=95, axis=0)
        median = np.percentile(gens, q=50, axis=0)
        plt.subplot(4, 3, m)
        plt.plot(ground_truth, color="orangered", label="Ground truth", linewidth=3)
        colors = ["lightgreen", "limegreen", "darkgreen"]
        plt.stackplot(np.arange(len(ground_truth)), lb_90, lb_50-lb_90, ub_50-lb_50, ub_90-ub_50,
                      colors=["white", colors[0], colors[1], colors[0]], baseline="zero")
        plt.plot(median, color=colors[2], label="Prediction", linewidth=3)
        plt.plot([], [], color="lightgreen", label="90% PI", linewidth=6)
        plt.plot([], [], color="limegreen", label="50% PI", linewidth=6)
        plt.legend(prop={'size': 10}, loc="upper left", frameon=False)
        y_max = np.max(ub_90)
        if y_max % 20 != 0:
            y_max = (y_max//20+1)*20
        config_time_axis(2, y_max)
        plt.title(day, fontsize=18)
    # fig.text(0.5, -0, 'Time [hour]', ha='center', fontsize=20)
    # fig.text(0, 0.5, 'Charging load [kW]', va='center', rotation='vertical', fontsize=20)
    plt.tight_layout()
    plt.savefig(f"Figure/overall_pi.png")
    plt.clf()

def show_compare_pi1(days=["2019-10-8", "2019-12-13"]):
    plt.figure(figsize=(10, 5), dpi=200)
    for i in range(len(days)):
        day = days[i]
        pkl_path0 = f"generation/refine/0.001-partial-epoch21/{day}.pkl"
        with open(pkl_path0, "rb") as f:
            input_data0 = pkl.load(f)
        pkl_path1 = f"generation/without refine-epoch194/{day}.pkl"
        with open(pkl_path1, "rb") as f:
            input_data1 = pkl.load(f)
        ground_truth = input_data0["true"]
        gens0 = input_data0["generation"]
        gens1 = input_data1["generation"]
        lb0_50 = np.percentile(gens0, q=25, axis=0)
        ub0_50 = np.percentile(gens0, q=75, axis=0)
        lb0_90 = np.percentile(gens0, q=5, axis=0)
        ub0_90 = np.percentile(gens0, q=95, axis=0)
        median0 = np.percentile(gens0, q=50, axis=0)
        lb1_50 = np.percentile(gens1, q=25, axis=0)
        ub1_50 = np.percentile(gens1, q=75, axis=0)
        lb1_90 = np.percentile(gens1, q=5, axis=0)
        ub1_90 = np.percentile(gens1, q=95, axis=0)
        median1 = np.percentile(gens1, q=50, axis=0)
        plt.subplot(1, 2, i+1)
        plt.plot(ground_truth, color="orangered", label="Ground truth", linewidth=3)
        colors1 = ["moccasin", "orange", "darkorange"]
        plt.stackplot(np.arange(len(ground_truth)), lb1_90, lb1_50 - lb1_90, ub1_50 - lb1_50, ub1_90 - ub1_50,
                      colors=["white", colors1[0], colors1[1], colors1[0]], baseline="zero")
        plt.plot(median1, color=colors1[2], label="Without fine-tuning", linewidth=3)
        colors0 = ["lightgreen", "limegreen", "darkgreen"]
        plt.stackplot(np.arange(len(ground_truth)), lb0_90, lb0_50-lb0_90, ub0_50-lb0_50, ub0_90-ub0_50,
                      colors=["white", colors0[0], colors0[1], colors0[0]], baseline="zero")
        plt.plot(median0, color=colors0[2], label="With fine-tuning", linewidth=3)
        plt.legend(prop={'size': 10}, loc="upper left", frameon=True)
        y_max = np.max(ub0_90)
        if y_max % 20 != 0:
            y_max = (y_max // 20 + 1) * 20
        config_time_axis(2, y_max)
        plt.title(day, fontsize=18)
    plt.tight_layout()
    plt.savefig(f"Figure/compare_pi.png")
    plt.clf()

def show_compare_pi2(days=["2019-2-10", "2019-8-9"]):
    plt.figure(figsize=(10, 5), dpi=200)
    for i in range(len(days)):
        day = days[i]
        pkl_path0 = f"generation/refine/0.001-partial-epoch21/{day}.pkl"
        with open(pkl_path0, "rb") as f:
            input_data0 = pkl.load(f)
        pkl_path1 = f"generation/regression/{day}.pkl"
        with open(pkl_path1, "rb") as f:
            input_data1 = pkl.load(f)
        ground_truth = input_data0["true"]
        gens0 = input_data0["generation"]
        lb0_50 = np.percentile(gens0, q=25, axis=0)
        ub0_50 = np.percentile(gens0, q=75, axis=0)
        lb0_90 = np.percentile(gens0, q=5, axis=0)
        ub0_90 = np.percentile(gens0, q=95, axis=0)
        median0 = np.percentile(gens0, q=50, axis=0)
        lb1_50 = input_data1[0.25]
        ub1_50 = input_data1[0.75]
        lb1_90 = input_data1[0.05]
        ub1_90 = input_data1[0.95]
        median1 = input_data1[0.5]
        plt.subplot(1, 2, i+1)
        plt.plot(ground_truth, color="orangered", label="Ground truth", linewidth=3)
        colors1 = ["lightblue", "deepskyblue", "blue"]
        plt.stackplot(np.arange(len(ground_truth)), lb1_90, lb1_50 - lb1_90, ub1_50 - lb1_50, ub1_90 - ub1_50,
                      colors=["white", colors1[0], colors1[1], colors1[0]], baseline="zero")
        plt.plot(median1, color=colors1[2], label="Quantile regression", linewidth=3)
        colors0 = ["lightgreen", "limegreen", "darkgreen"]
        plt.stackplot(np.arange(len(ground_truth)), lb0_90, lb0_50-lb0_90, ub0_50-lb0_50, ub0_90-ub0_50,
                      colors=["white", colors0[0], colors0[1], colors0[0]], baseline="zero")
        plt.plot(median0, color=colors0[2], label="DiffPLF", linewidth=3)
        plt.legend(prop={'size': 10}, loc="upper left", frameon=True)
        y_max = np.max(ub1_90)
        if y_max % 20 != 0:
            y_max = (y_max // 20 + 1) * 20
        config_time_axis(2, y_max)
        plt.title(day, fontsize=18)
    plt.tight_layout()
    plt.savefig(f"Figure/contrast_pi.png")
    plt.clf()

def show_overall_metrics():
    x = np.arange(0, 1+1, 1)
    width = 0.25
    err_attr = {"elinewidth": 2, "ecolor": "black", "capsize": 0}  # 误差棒属性
    diff_mae_mean, diff_mae_std = 7.161, 1.5569
    diff_crps_mean, diff_crps_std = 5.067, 1.0943
    reg_mae_mean, reg_mae_std = 11.8518, 3.936
    reg_crps_mean, reg_crps_std = 10.1071, 2.1239
    rects1 = plt.bar(x-width/2, [diff_mae_mean, reg_mae_mean], yerr=[diff_mae_std, reg_mae_std], error_kw=err_attr,
                     width=width, label='MAE', color="#e377c2")
    rects2 = plt.bar(x+width/2, [diff_crps_mean, reg_crps_mean], yerr=[diff_crps_std, reg_crps_std], error_kw=err_attr,
                     width=width, label='CRPS', color="#8c564b")
    fs = 12
    plt.legend(loc="upper left", fontsize=fs)
    plt.xlabel(f"Metrics", fontsize=fs)
    plt.ylabel(f"Value", fontsize=fs)
    plt.xticks(x, ["DiffPLF", "Quantile Regression"], fontsize=fs)
    plt.ylim((0, 16))
    plt.tight_layout()
    plt.savefig(f"figure/overall_metric.png")
    plt.clf()

def show_varing_horizons():
    dir = "generation/prediction horizon"
    # input_files = {"12h-epoch13": ["2019-1-6", "2019-2-27", "2019-9-10"],
    #                "6h-epoch16": ["2019-1-20", "2019-10-14", "2019-12-22"],
    #                "4h-epoch83": ["2019-6-18", "2019-10-6", "2019-11-4"],
    #                "1h-epoch28": ["2019-2-11", "2019-10-18", "2019-12-18"]
    #                }
    input_files = {"12h-epoch13": ["2019-1-6"],
                   "6h-epoch16": ["2019-1-20"],
                   "4h-epoch83": ["2019-6-18"],
                   "1h-epoch28": ["2019-2-11"]
                   }
    offset = [(24-12)*4, (24-6)*4, (24-4)*4, (24-1)*4]
    i = 1
    plt.figure(figsize=(11, 9), dpi=200)
    for folder in input_files.keys():
        for day in input_files[folder]:
            dev = offset[(i-1)//1]
            pkl_path = f"{dir}/{folder}/{day}.pkl"
            with open(pkl_path, "rb") as f:
                input_data = pkl.load(f)
            with open(f"dataset/daily/{day}.pkl", "rb") as f:
                obs = pkl.load(f)["profile"][0:dev]
            ground_truth = input_data["true"]
            gens = input_data["generation"]
            lb_50 = np.percentile(gens, q=25, axis=0)
            ub_50 = np.percentile(gens, q=75, axis=0)
            lb_90 = np.percentile(gens, q=5, axis=0)
            ub_90 = np.percentile(gens, q=95, axis=0)
            median = np.percentile(gens, q=50, axis=0)
            xt = np.arange(len(ground_truth)) + dev
            plt.subplot(2, 2, i)
            plt.plot(np.arange(dev+1), obs.tolist()+[ground_truth[0]], color="dodgerblue", label="Observation", linewidth=1.5)
            colors = ["lightgreen", "limegreen", "darkgreen"]
            plt.stackplot(xt, lb_90, lb_50 - lb_90, ub_50 - lb_50, ub_90 - ub_50,
                          colors=["white", colors[0], colors[1], colors[0]], baseline="zero")
            plt.plot(xt, median, color=colors[2], label="Prediction", linewidth=1.5, marker=".", markersize=4)
            plt.plot(xt, ground_truth, color="orangered", label="Ground truth", linewidth=1.5, marker="x", markersize=3)
            plt.plot([], [], color="lightgreen", label="90% PI", linewidth=6)
            plt.plot([], [], color="limegreen", label="50% PI", linewidth=6)
            plt.legend(prop={'size': 10}, loc="upper right", frameon=False)
            y_max = max(np.max(ub_90), np.max(obs))
            if y_max % 20 != 0:
                y_max = (y_max // 20 + 1) * 20
            config_time_axis(2, y_max)
            plt.title(f"{folder.split('-')[0][0:-1]} hour: {day}", fontsize=18)
            plt.grid(ls="--")
            i += 1
    plt.tight_layout()
    plt.savefig(f"Figure/horizon.png")
    plt.clf()

def show_add_metrics():
    plt.figure(figsize=(8, 4.5), dpi=200)
    x = np.arange(0, 1+1, 1)
    width = 0.20
    err_attr = {"elinewidth": 1.5, "ecolor": "black", "capsize": 0}  # 误差棒属性
    fs = 15
    # varying horizons
    plt.subplot(1, 2, 1)
    plt.grid(linestyle='--')
    h12_mae_mean, h12_mae_std = 7.670, 1.461
    h12_crps_mean, h12_crps_std = 5.431, 0.978
    h6_mae_mean, h6_mae_std = 5.240, 2.045
    h6_crps_mean, h6_crps_std = 3.719, 1.416
    h4_mae_mean, h4_mae_std = 3.467, 1.714
    h4_crps_mean, h4_crps_std = 2.4785, 1.173
    h1_mae_mean, h1_mae_std = 1.000, 0.939
    h1_crps_mean, h1_crps_std = 0.742, 0.629
    rects1 = plt.bar(x-width*1.5, [h1_mae_mean, h1_crps_mean], yerr=[h1_mae_std, h1_crps_std], error_kw=err_attr,
                     width=width, label='1 hour', color="#fdeeba")
    rects2 = plt.bar(x-width*0.5, [h4_mae_mean, h4_crps_mean], yerr=[h4_mae_std, h4_crps_std], error_kw=err_attr,
                      width=width, label='4 hour', color="#f7e16f")
    rects3 = plt.bar(x+width*0.5, [h6_mae_mean, h6_crps_mean], yerr=[h6_mae_std, h6_crps_std], error_kw=err_attr,
                     width=width, label='6 hour', color="#f0b76f")
    rects4 = plt.bar(x+width*1.5, [h12_mae_mean, h12_crps_mean], yerr=[h12_mae_std, h12_crps_std], error_kw=err_attr,
                     width=width, label='12 hour', color="#e59c58")
    plt.legend(loc="upper right", fontsize=fs-4)
    plt.xlabel(f"Metrics\n\n(a) Prediction horizon", fontsize=fs+2)
    plt.ylabel(f"Value", fontsize=fs+2)
    plt.xticks(x, ["MAE", "CRPS"], fontsize=fs)
    plt.ylim((0, 10))
    plt.yticks(fontsize=fs)
    # EV number deviation
    plt.subplot(1, 2, 2)
    plt.grid(linestyle='--')
    v5p_mae_mean, v5p_mae_std = 7.550, 1.546
    v5p_crps_mean, v5p_crps_std = 5.344, 1.094
    v5n_mae_mean, v5n_mae_std = 7.279, 1.600
    v5n_crps_mean, v5n_crps_std = 5.145, 1.146
    v10p_mae_mean, v10p_mae_std = 7.925, 1.676
    v10p_crps_mean, v10p_crps_std = 5.620, 1.195
    v10n_mae_mean, v10n_mae_std = 9.507, 2.127
    v10n_crps_mean, v10n_crps_std = 6.741, 1.595
    rects1 = plt.bar(x-width*1.5, [v5p_mae_mean, v5p_crps_mean], yerr=[v5p_mae_std, v5p_crps_std], error_kw=err_attr,
                     width=width, label='+5%', color="#e8c6e1")
    rects2 = plt.bar(x-width*0.5, [v5n_mae_mean, v5n_crps_mean], yerr=[v5n_mae_std, v5n_crps_std], error_kw=err_attr,
                     width=width, label='-5%', color="#d4bddc")
    rects3 = plt.bar(x+width*0.5, [v10p_mae_mean, v10p_crps_mean], yerr=[v10p_mae_std, v10p_crps_std], error_kw=err_attr,
                     width=width, label='+10%', color="#b28cbf")
    rects4 = plt.bar(x+width*1.5, [v10n_mae_mean, v10n_crps_mean], yerr=[v10n_mae_std, v10n_crps_std],
                     error_kw=err_attr, width=width, label='-10%', color="#ab6ca0")
    plt.legend(loc="upper right", fontsize=fs-4)
    plt.xlabel(f"Metrics\n\n(b) EV number deviation", fontsize=fs+2)
    plt.ylabel(f"Value", fontsize=fs+2)
    plt.xticks(x, ["MAE", "CRPS"], fontsize=fs)
    plt.ylim((0, 12))
    plt.yticks(fontsize=fs)
    plt.tight_layout()
    plt.savefig(f"figure/add metric.png")
    plt.clf()

def power_to_energy(ps):  # kW to kWh
    piecewise_energy = []
    for i in range(len(ps)):
        if i == 0:
            e = ps[i] / 8
        else:
            e = (ps[i - 1] + ps[i]) / 8
        piecewise_energy.append(e)
    piecewise_energy = np.cumsum(piecewise_energy)
    return piecewise_energy

def show_ev_num(days):
    plt.figure(figsize=(5, 4), dpi=200)
    devs = [0.2, 0.6, 1.0, 1.4, 2.0]
    light_colors = ["lavender", "paleturquoise", "palegreen", "moccasin", "lightblue"]
    dark_colors = ["purple", "darkturquoise", "darkgreen", "orange", "steelblue"]
    sample_num = 100
    for d in days:
        for i in range(len(devs)):
            folder = f"generation/EV number/{d}/{np.around(devs[i], 1)}"
            pkl_path = f"{folder}/{d}.pkl"
            with open(pkl_path, "rb") as f:
                gen_data = pkl.load(f)
            ground_truth = gen_data["true"]
            gen_samples = np.array(gen_data["generation"])
            mae = np.mean(np.abs(gen_samples-ground_truth), axis=1)
            if devs[i] == 1.0:
                low_indexes = np.argsort(mae)[0:sample_num]
                gen_samples = gen_samples[low_indexes]
            else:
                high_indexes = np.argsort(mae)[-sample_num:]
                gen_samples = gen_samples[high_indexes]
            # gen_samples = np.cumsum(gen_samples, axis=1)
            energy_samples = []
            for j in range(len(gen_samples)):
                piecewise_energy = power_to_energy(gen_samples[j])
                if j == 0:
                    plt.plot(piecewise_energy, color=light_colors[i], linewidth=0.3)
                else:
                    plt.plot(piecewise_energy, color=light_colors[i], linewidth=0.3)
                energy_samples.append(piecewise_energy)
            plt.plot(np.mean(energy_samples, axis=0), color=dark_colors[i], linewidth=1.2, label=f"{int(devs[i]*115)} EV")
        ground_truth = power_to_energy(ground_truth)
        plt.plot(ground_truth, c="red", linewidth=1.2, linestyle="--", label="Ground truth (115 EV)")
        plt.legend()
    plt.title(f"{days[0]}", fontsize=15)
    y_max = np.max(energy_samples)
    config_time_axis(2, y_max)
    plt.xlabel("Time [hour]", fontsize=15)
    plt.ylabel("Cumulative charging energy [kWh]", fontsize=15)
    plt.grid(axis='y', linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"figure/ev number.png")
    plt.clf()

def plot_profile():
    np.random.seed(10)
    noise = np.random.normal(0, 1, 96)
    # with open("generation/refine/0.01-partial-epoch74/2019-1-10.pkl", "rb") as f:
    #     profile = np.array(pkl.load(f)["generation"])
    # # profile /= np.max(profile)
    # # xt = (0.992**0.5)*profile + (0.008**0.5)*noise
    # fig = plt.figure(figsize=(6, 4))
    # fig.set_tight_layout(True)
    plt.plot(noise, linewidth=6, color="black")
    # for p in profile[-5:]:
    #     plt.plot(p, linewidth=3, color="green")
    plt.axis('off')
    plt.savefig("figure/sample/noise.png")

if __name__ == "__main__":
    random_days = [6, 11, 27, 10, 3, 21, 13, 25, 12, 9, 20, 18]
    # show_overall_pi("generation/refine/0.001-partial-epoch21", random_days)
    # show_overall_metrics()
    # show_compare_pi1()
    show_compare_pi2()
    # show_varing_horizons()
    # show_add_metrics()
    # show_ev_num(["2019-6-8"])
    # plot_profile()
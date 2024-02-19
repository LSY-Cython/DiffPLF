import numpy as np
import os
import pickle as pkl
import CRPS.CRPS as pscore
import properscoring as ps

def plf_eval(input_folder):
    gen_paths = [f"{input_folder}/{f}" for f in os.listdir(input_folder) if f.endswith("pkl")]
    daily_results = {}
    mae = []
    rmse = []
    crps = []
    for gp in gen_paths:
        # # select a specific day for comparison
        # day = "2019-1-10"
        # if day not in gp:
        #     continue
        with open(gp, "rb") as f:
            gen_data = pkl.load(f)
        ground_truth = gen_data["true"]
        point_pred = gen_data["median"]
        gen_samples = gen_data["generation"]
        # deterministic prediction performance
        error = np.mean(np.abs(ground_truth-point_pred))
        mae.append(error)
        error = np.sqrt(np.mean(np.square(ground_truth-point_pred)))
        rmse.append(error)
        # probabilistic forecast performance
        crps_t = []
        pred_len = np.array(gen_samples).shape[1]
        for t in range(pred_len):
            xt = gen_samples[:, t]
            st, _, _ = pscore(xt, ground_truth[t]).compute()
            crps_t.append(st)
        # avaerage over the prediction horizon
        crps.append(np.mean(crps_t))
        day = gp.split("/")[-1].rstrip(".pkl")
        daily_results[day] = [mae[-1], crps[-1]]
    # # select some days for comparison
    # low_indices = np.argsort(crps)[0:200]
    # mae = np.array(mae)[low_indices]
    # rmse = np.array(rmse)[low_indices]
    # crps = np.array(crps)[low_indices]
    print(f"MAE: {np.around(np.mean(mae), 4)}±{np.around(np.std(mae), 4)}")
    print(f"RMSE: {np.around(np.mean(rmse), 4)}±{np.around(np.std(rmse), 4)}")
    print(f"CRPS: {np.around(np.mean(crps), 4)}±{np.around(np.std(crps), 4)}")
    return daily_results

def reg_eval(input_folder):
    reg_paths = [f"{input_folder}/{f}" for f in os.listdir(input_folder) if f.endswith("pkl")]
    mae = []
    rmse = []
    crps = []
    for rp in reg_paths:
        with open(rp, "rb") as f:
            reg_data = pkl.load(f)
        ground_truth = reg_data["true"]
        point_pred = reg_data[0.5]
        ub_50 = reg_data[0.95]
        # deterministic prediction performance
        error = np.mean(np.abs(ground_truth - point_pred))
        mae.append(error)
        error = np.sqrt(np.mean(np.square(ground_truth - point_pred)))
        rmse.append(error)
        # probabilistic forecast performance
        crps_t = []
        for i in range(len(ground_truth)):
            point = ground_truth[i]
            mu = point_pred[i]
            sig = np.abs(ub_50[i] - point_pred[i])
            score = ps.crps_gaussian(point, mu=mu, sig=sig)
            crps_t.append(score)
        crps.append(np.mean(crps_t))
    print(f"MAE: {np.around(np.mean(mae), 4)}±{np.around(np.std(mae), 4)}")
    print(f"RMSE: {np.around(np.mean(rmse), 4)}±{np.around(np.std(rmse), 4)}")
    print(f"CRPS: {np.around(np.mean(crps), 4)}±{np.around(np.std(crps), 4)}")

if __name__ == "__main__":
    # r1 = plf_eval("generation/without refine-epoch194")
    # r2 = plf_eval("generation/refine/0.001-partial-epoch21")
    # metric_error = dict()
    # for day in r1.keys():
    #     v1 = r1[day]
    #     v2 = r2[day]
    #     metric_error[day] = [v1[0]-v2[0], v1[1]-v2[1]]

    plf_eval("generation")

    # plf_eval("generation/no cross-epoch186")
    # plf_eval("generation/refine/0.001-whole-epoch24")
    # plf_eval("generation/refine/no selection")
    # plf_eval("generation/pretrain-no selection")
    # plf_eval("generation/no constrain-epoch188")
    # plf_eval("generation/EV number/-10%")
    # plf_eval("generation/EV number/+10%")
    # plf_eval("generation/EV number/-5%")
    # plf_eval("generation/EV number/+5%")

    # r_h12 = plf_eval("generation/prediction horizon/12h-epoch13")
    # r_h6 = plf_eval("generation/prediction horizon/6h-epoch16")
    # r_h4 = plf_eval("generation/prediction horizon/4h-epoch83")
    # r_h1 = plf_eval("generation/prediction horizon/1h-epoch28")
    # print(r_h1)
    # reg_eval("generation/regression")
    pass
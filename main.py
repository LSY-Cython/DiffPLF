from diffusion import *
from dataset import creat_dataloader
from options import Options
import sys
import json
import shutil

if __name__ == "__main__":
    # 1) pre-training 2) collect point forecast outcomes 3) fine-tuning
    # triple false refers to test
    isTrain = False
    isRefine = False
    isCollect = False  # collect median samples for fine-tuning
    pred_horizon = 96
    opt = Options(isTrain, isRefine)
    modules = sys.modules
    data_file = f"dataset/pred_dataset.json"
    data_loader = creat_dataloader(data_file, opt.batch_size, opt.shuffle, isRefine, opt.isConstrain)
    model = cDDPM(opt, data_loader)
    if isTrain:
        model.train()
    elif isRefine:
        init_epoch = 194
        model.fine_tuning(f"weights/train/{opt.weight_sign}/epoch{init_epoch}.pt",
                          pred_len=pred_horizon  # to vary the prediction horizon, normally 96(24h)
                          )
    else:
        best_epoch = 194  # base model
        # best_epoch = 21  # refined model
        if isCollect:
            dir = "train"
            key = "train"
        else:
            dir = "refine"
            key = "test"
        with open(data_file, "r") as f:
            test_paths = json.load(f)[key]
        # if os.path.exists("generation"):
        #     shutil.rmtree("generation")
        #     os.makedirs("generation")
        for test_file in test_paths:
        # days = ["dataset/daily/2019-10-8.pkl", "dataset/daily/2019-12-13.pkl", "dataset/daily/2019-1-6.pkl",
        #         "dataset/daily/2019-2-11.pkl", "dataset/daily/2019-3-27.pkl", "dataset/daily/2019-4-10.pkl",
        #         "dataset/daily/2019-5-3.pkl", "dataset/daily/2019-6-21.pkl", "dataset/daily/2019-7-13.pkl",
        #         "dataset/daily/2019-8-25.pkl", "dataset/daily/2019-9-12.pkl", "dataset/daily/2019-10-9.pkl",
        #         "dataset/daily/2019-11-20.pkl", "dataset/daily/2019-12-18.pkl"]
        # days = ["dataset/daily/2019-2-10.pkl", "dataset/daily/2019-8-9.pkl"]
        # for test_file in days:  # 115
        #     dir = f"train/{opt.weight_sign}"  # test after 1st-stage training
            model.infer(weight_path=f"weights/{dir}/epoch{best_epoch}.pt",
                        n_samples=200,  # default 1000
                        test_file=test_file,
                        isCollect=isCollect,
                        ev_ratio=1,  # to control EV number, normally 1
                        pred_len=pred_horizon
                        )
        print(f"Average inference time: {np.mean(model.infer_time)}s/case")
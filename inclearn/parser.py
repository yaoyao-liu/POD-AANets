import argparse

def get_parser():
    parser = argparse.ArgumentParser("IncLearner", description="Incremental Learning trainer.")
    parser.add_argument("-m", "--model", default="icarl", type=str)
    parser.add_argument("-c", "--convnet", default="rebuffi", type=str)
    parser.add_argument("--dropout", default=0., type=float)
    parser.add_argument("-he", "--herding", default=None, type=str)
    parser.add_argument("-memory", "--memory-size", default=2000, type=int)
    parser.add_argument("-temp", "--temperature", default=1, type=int)
    parser.add_argument("-fixed-memory", "--fixed-memory", action="store_true")
    parser.add_argument("-disable-rmm", "--disable-rmm", action="store_true")
    parser.add_argument("-disable-search", "--disable-search", action="store_true")
    parser.add_argument("-d", "--dataset", default="cifar100", type=str)
    parser.add_argument("-inc", "--increment", default=20, type=int)
    parser.add_argument("-b", "--batch-size", default=128, type=int)
    parser.add_argument("-w", "--workers", default=0, type=int)
    parser.add_argument("-t", "--threads", default=1, type=int)
    parser.add_argument("-v", "--validation", default=0., type=float)
    parser.add_argument("-random", "--random-classes", action="store_true", default=False)
    parser.add_argument("-order", "--order")
    parser.add_argument("-mem_rate_list", "--mem_rate_list")
    parser.add_argument("-cls_rate_list", "--cls_rate_list")
    parser.add_argument("-max-task", "--max-task", default=None, type=int)
    parser.add_argument("-onehot", "--onehot", action="store_true")
    parser.add_argument("-initial-increment", "--initial-increment", default=None, type=int)
    parser.add_argument("-sampler", "--sampler")
    parser.add_argument("-data-path", "--data-path", default="./data/", type=str)
    parser.add_argument("-lr", "--lr", default=2., type=float)
    parser.add_argument("-wd", "--weight-decay", default=0.00005, type=float)
    parser.add_argument("-sc", "--scheduling", default=[49, 63], nargs="*", type=int)
    parser.add_argument("-lr-decay", "--lr-decay", default=1/5, type=float)
    parser.add_argument("-opt", "--optimizer", default="sgd", type=str)
    parser.add_argument("-e", "--epochs", default=70, type=int)
    parser.add_argument("--debug_mode", action="store_true", default=False)
    parser.add_argument("--device", default=[0], type=int, nargs="+")
    parser.add_argument("--label", type=str)
    parser.add_argument("--autolabel", action="store_true")
    parser.add_argument("-seed", "--seed", default=[1], type=int, nargs="+")
    parser.add_argument("-seed-range", "--seed-range", type=int, nargs=2)
    parser.add_argument("-options", "--options", nargs="+")
    parser.add_argument("-save", "--save-model", choices=["never", "last", "task", "first"], default="task")
    parser.add_argument("--dump-predictions", default=False, action="store_true")
    parser.add_argument("-log", "--logging", choices=["critical", "warning", "info", "debug"], default="info")
    parser.add_argument("-resume", "--resume", default=None)
    parser.add_argument("-resume_first_ckpt", "--resume_first_ckpt", default=None)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--resume_first", action="store_true", default=False)
    parser.add_argument("--using_compressed_exemplar", action="store_true", default=False)
    parser.add_argument("--recompute-meta", action="store_true", default=False)
    parser.add_argument("--no-benchmark", action="store_true", default=False)
    parser.add_argument("--detect-anomaly", action="store_true", default=False)
    parser.add_argument("--weight_pod_loss", default=1.0, type=float)
    parser.add_argument("--weight_icarl_loss", default=0.0, type=float)
    parser.add_argument("--weight_lucir_loss", default=0.0, type=float)

    return parser
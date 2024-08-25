import argparse

def parse_args(cfg):
    parser = argparse.ArgumentParser(description='Hello')
    parser.add_argument('--run_mode', type=str, default='test_paper', help='run mode (train, test, test_paper)')
    parser.add_argument('--pre_dir', type=str, default='--root/preprocessed/DATASET_NAME/', help='preprocessed data dir')
    parser.add_argument('--dataset_dir', default=None, help='dataset dir')
    parser.add_argument('--paper_weight_dir', default='--root/pretrained/DATASET_NAME/', help='pretrained weights dir (paper)')
    args = parser.parse_args()

    cfg = args_to_config(cfg, args)
    return cfg

# Example on my PC env
# --------------------------------------------------------
def parse_args_default(cfg):
    root = '/media/dkjin/4fefb28c-5de9-4abd-a935-aa2d61392048'
    parser = argparse.ArgumentParser(description='Hello')
    parser.add_argument('--run_mode', type=str, default=None, help='run mode (train, test, test_paper)')
    parser.add_argument('--pre_dir', type=str, default=f'{root}/Work/Current/Lane_detection/Project_02/P07_github/preprocessed/VIL-100', help='preprocessed data dir')
    parser.add_argument('--dataset_dir', default='/home/dkjin/Project/Dataset/VIL-100', help='dataset')
    parser.add_argument('--paper_weight_dir', default=f'{root}/Work/Current/Lane_detection/Project_02/P07_github/pretrained/VIL-100', help='pretrained weights dir (paper)')
    args = parser.parse_args()

    cfg = args_to_config(cfg, args)
    return cfg
# --------------------------------------------------------

def args_to_config(cfg, args):
    if args.dataset_dir is not None:
        cfg.dir['dataset'] = args.dataset_dir
    if args.pre_dir is not None:
        cfg.dir['head_pre'] = args.pre_dir
        cfg.dir['pre0_train'] = cfg.dir['pre0_train'].replace('--preprocessed data path', args.pre_dir)
        cfg.dir['pre0_test'] = cfg.dir['pre0_test'].replace('--preprocessed data path', args.pre_dir)
        cfg.dir['pre2'] = cfg.dir['pre2'].replace('--preprocessed data path', args.pre_dir)
        cfg.dir['pre3_train'] = cfg.dir['pre3_train'].replace('--preprocessed data path', args.pre_dir)
        cfg.dir['pre3_test'] = cfg.dir['pre3_test'].replace('--preprocessed data path', args.pre_dir)
        cfg.dir['weight_paper'] = cfg.dir['weight_paper'].replace('--pretrained data path', args.paper_weight_dir)

    if args.run_mode is not None:
        cfg.run_mode = args.run_mode

    if args.run_mode == 'test_paper':
        cfg.do_eval_iou = True
        cfg.do_eval_temporal = True
        cfg.do_eval_iou_laneatt = False
        cfg.sampling = False
        cfg.dir['weight_paper'] = args.paper_weight_dir


    return cfg
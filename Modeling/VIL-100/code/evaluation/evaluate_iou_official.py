import os
import platform
from skimage.io import imread

def read_helper(path):
    lines = open(path, 'r').readlines()[1:]
    lines = ' '.join(lines)
    values = lines.split(' ')[1::2]
    keys = lines.split(' ')[0::2]
    keys = [key[:-1] for key in keys]
    res = {k : v for k,v in zip(keys,values)}
    return res

class LaneEval_CULane(object):
    def __init__(self, cfg=None):
        self.cfg = cfg

    def call_culane_eval(self, data_dir, output_path, temp_dir, result_dir):
        detect_dir = f'{self.cfg.dir["out"]}/{self.mode}/results/iou/txt/pred_txt'         
        w_lane = 30
        iou = self.iou  # Set iou to 0.5 or 0.8
        frame = 1
        list = os.path.join(data_dir, 'data', 'test.txt')
        file = open(list)
        if not os.path.exists(os.path.join(output_path, temp_dir)):
            os.makedirs(os.path.join(output_path, temp_dir))
        if not os.path.exists(os.path.join(output_path, result_dir)):
            os.mkdir(os.path.join(output_path, result_dir))

        for line in file.readlines(): #[save_freq:]
            txt_path = os.path.join(output_path, temp_dir, line.strip().split('/')[2]+'.txt')        
            with open(txt_path, "a+") as f:
                f.write(line.strip().replace('/JPEGImages','') + '\n')
            f.close()

        eval_cmd = f'{os.getcwd()}/evaluation/culane/evaluate'
        if platform.system() == 'Windows':
            eval_cmd = eval_cmd.replace('/', os.sep)
        list_test_files = os.listdir(os.path.join(output_path, temp_dir))
        res_all = {}
        for list_file in list_test_files:
            txt_name = os.path.join(output_path, temp_dir, list_file)

            with open(txt_name, 'r') as fp:
                frame_path = os.path.join(data_dir, 'JPEGImages', fp.readlines()[0][1:].strip())
                frame1 = imread(frame_path, as_gray=True)
            fp.close()

            out0 = os.path.join(output_path, result_dir, list_file)
            ano_dir_temp = f'{data_dir}/txt/anno_txt'
            img_dir_temp = os.path.join(data_dir, 'JPEGImages')

            os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s' % (
            eval_cmd, ano_dir_temp, detect_dir, img_dir_temp, txt_name, w_lane, iou, frame1.shape[1], frame1.shape[0], frame, out0))
            res_all['out_'+str(list_file[:-4])] = read_helper(out0)
        return res_all

    def measure_IoU(self, mode, iou):
        print('------------------start calculate F and mIoU---------------------')
        self.mode = mode
        self.iou = iou
        data_root = self.cfg.dir['dataset']        
        work_dir = f'{self.cfg.dir["out"]}/{mode}/results/iou/txt/results_txt'    
        temp_dir = 'temp'
        result_dir = 'results'
        res = self.call_culane_eval(data_root, work_dir, temp_dir, result_dir)
        TP, FP, FN, MIOU = 0, 0, 0, 0
        for k, v in res.items():
            val = float(v['Fmeasure']) if ('nan' or '-nan') not in v['Fmeasure'] else 0
            val_tp, val_fp, val_fn, val_iou = int(v['tp']), int(v['fp']), int(v['fn']), float(v['miou'])
            TP += val_tp
            FP += val_fp
            FN += val_fn
            MIOU += val_iou
            # dist_print(k, val)
        P = TP * 1.0 / (TP + FP)
        R = TP * 1.0 / (TP + FN)
        F = 0.0 if (P + R) == 0 else 2 * P * R / (P + R)
        print('all scenes F:', F)
        print('all scenes miou:', MIOU / len(res.items()))

        return {'p': P, 'r': R, 'F1': F, 'miou': MIOU / len(res.items())}

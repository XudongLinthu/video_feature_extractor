import torch as th
import math
import os, shutil, time
import numpy as np
from video_loader import VideoLoader
from torch.utils.data import DataLoader
import argparse
from model import get_model
from functools import partial

from preprocessing import Preprocessing
from random_sequence_shuffler import RandomSequenceSampler
import torch.nn.functional as F

def extract(args, csv):

    dataset = VideoLoader(
        csv,
        framerate=1 if args.type == '2d' else 24,
        size=224 if args.type != '3d' else 112,
        centercrop=(args.type != '2d'),
    )
    n_dataset = len(dataset)
    sampler = RandomSequenceSampler(n_dataset, 10)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_decoding_thread,
        sampler=sampler if n_dataset > 10 else None,
    )
    preprocess = Preprocessing(args.type)
    model = get_model(args)

    with th.no_grad():
        for k, data in enumerate(loader):
            input_file = data['input'][0]
            output_file = data['output'][0]
            if len(data['video'].shape) > 3:
                print('Computing features of video {}/{}: {}'.format(
                    k + 1, n_dataset, input_file))
                video = data['video'].squeeze()
                if len(video.shape) == 4:
                    video = preprocess(video)
                    n_chunk = len(video)
                    features = th.cuda.FloatTensor(n_chunk, 2048).fill_(0)
                    n_iter = int(math.ceil(n_chunk / float(args.batch_size)))
                    for i in range(n_iter):
                        min_ind = i * args.batch_size
                        max_ind = (i + 1) * args.batch_size
                        video_batch = video[min_ind:max_ind].cuda()
                        if args.type == "ig":
                            video_batch = video_batch.permute(1, 0, 2, 3).unsqueeze(0)
                            batch_features = model(video_batch)
                        else:
                            batch_features = model(video_batch)
                        if args.l2_normalize:
                            batch_features = F.normalize(batch_features, dim=1)
                        features[min_ind:max_ind] = batch_features
                    features = features.cpu().numpy()
                    if args.half_precision:
                        features = features.astype('float16')
                    np.save(output_file, features)
            else:
                print('Video {} already processed.'.format(input_file))


parser = argparse.ArgumentParser(description='Easy video feature extractor')

parser.add_argument(
    '--csv',
    type=str,
    help='input csv with video input path')
parser.add_argument('--batch_size', type=int, default=64,
                            help='batch size')
parser.add_argument('--type', type=str, default='2d',
                            help='CNN type')
parser.add_argument('--half_precision', type=int, default=1,
                            help='output half precision float')
parser.add_argument('--num_decoding_thread', type=int, default=4,
                            help='Num parallel thread for video decoding')
parser.add_argument('--l2_normalize', type=int, default=1,
                            help='l2 normalize feature')
parser.add_argument('--resnext101_model_path', type=str, default='model/resnext101.pth',
                            help='Resnext model path')
parser.add_argument('--ig_model_path', type=str, default='/checkpoint/bkorbar/HowTo100M/R25D_ft_Kinetics.pth',
                            help='Resnext model path')
args = parser.parse_args()


if os.path.isfile(args.csv):
    extract(args, args.csv)
if args.type == 'ig':
    args.batch_size = 24

if os.path.isdir(args.csv):
    print("Some better message")
    import submitit
    log_dir = "/checkpoint/bkorbar/logs_{}".format(args.type)
    os.makedirs(log_dir, exist_ok=True)
    executor = submitit.AutoExecutor(folder=log_dir)
    executor.update_parameters(
        name="ig_fe_antoine",
        mem_gb=80 if args.type != "ig" else 230,
        gpus_per_node=1,
        cpus_per_task=10,
        timeout_min=60*12,
        # Below are cluster dependent parameters
        partition='learnfair',
        signal_delay_s=120,
    )
    k = [os.path.join(args.csv, fil) for fil in os.listdir(args.csv) if fil != '.ipynb_checkpoints']
    print(k)
    
    partial_fe = partial(extract, args)
    jobs = executor.map_array(partial_fe, k)

    num_finished = sum(job.done() for job in jobs)
    # wait and check how many have finished
    while num_finished < len(jobs):
        time.sleep(360)
        num_finished = sum(job.done() for job in jobs)
        print("Feature extraction:\n \t type: ", args.type, "\n \t finished: ", num_finished)

    # cleanup
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)

 
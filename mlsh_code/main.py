import argparse
import tensorflow as tf
parser = argparse.ArgumentParser()
parser.add_argument('savename', type=str)
parser.add_argument('--task', type=str)
parser.add_argument('--num_subs', type=int)
parser.add_argument('--macro_duration', type=int)
parser.add_argument('--num_rollouts', type=int)
parser.add_argument('--warmup_time', type=int)
parser.add_argument('--train_time', type=int)
parser.add_argument('--force_subpolicy', type=int)
parser.add_argument('--replay', type=str)
parser.add_argument('-s', action='store_true')
parser.add_argument('--continue_iter', type=str)
args = parser.parse_args()
sess = None

# python main.py --task MovementBandits-v0 --num_subs 2 --macro_duration 10 --num_rollouts 1000 --warmup_time 60 --train_time 1 --replay True test

from mpi4py import MPI
from rl_algs.common import set_global_seeds, tf_util as U
import os.path as osp
import gym, logging
import numpy as np
from collections import deque
from gym import spaces
import misc_util
import sys
import shutil
import subprocess
import master

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

replay = str2bool(args.replay)
args.replay = str2bool(args.replay)

RELPATH = osp.join(args.savename)
LOGDIR = osp.join('/root/results' if sys.platform.startswith('linux') else '/tmp', RELPATH)

def callback(it):
    global sess
    print("it: %s\n\n" % it)
    if MPI.COMM_WORLD.Get_rank()==0:
        if (it > 0) and (not replay):
            fname = osp.join("savedir/", 'checkpoints', '%.5i'%it)
            print("save_state %s" % fname)
            U.save_state(fname)
            #tf.train.write_graph(sess.graph_def, osp.join("savedir/", 'checkpoints'), 'train%.5i.pb' % it, as_text=False)

    if it == 0 and args.continue_iter is not None:
        fname = osp.join("savedir/"+args.savename+"/checkpoints/", str(args.continue_iter))
        U.load_state(fname)
        pass

def train():
    global sess
    num_timesteps=1e9
    seed = 1401
    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()
    workerseed = seed + 1000 * MPI.COMM_WORLD.Get_rank()
    rank = MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)

    # if rank != 0:
    #     logger.set_level(logger.DISABLED)
    # logger.log("rank %i" % MPI.COMM_WORLD.Get_rank())

    world_group = MPI.COMM_WORLD.Get_group()
    mygroup = rank % 10
    theta_group = world_group.Incl([x for x in range(MPI.COMM_WORLD.size) if (x % 10 == mygroup)])
    comm = MPI.COMM_WORLD.Create(theta_group)
    comm.Barrier()
    # comm = MPI.COMM_WORLD

    master.start(callback, args=args, workerseed=workerseed, rank=rank, comm=comm)

def main():
    if MPI.COMM_WORLD.Get_rank() == 0 and osp.exists(LOGDIR):
        shutil.rmtree(LOGDIR)
    MPI.COMM_WORLD.Barrier()
    # with logger.session(dir=LOGDIR):
    train()

if __name__ == '__main__':
    main()

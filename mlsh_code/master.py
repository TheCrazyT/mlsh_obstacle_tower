import gym
import test_envs
import tensorflow as tf
import rollouts
from policy_network import Policy
from subpolicy_network import SubPolicy
from observation_network import Features
from guess_steps import GuessStepsPolicy
from learner import Learner
import rl_algs.common.tf_util as U
import numpy as np
# from tinkerbell import logger
import pickle

def start(callback, args, workerseed, rank, comm):
    env = gym.make(args.task)
    env.seed(workerseed)
    np.random.seed(workerseed)
    ob_space = env.observation_space
    ac_space = env.action_space
    print("ob_space: %s" % ob_space)
    print("ac_space: %s" % ac_space)

    num_subs = args.num_subs
    macro_duration = args.macro_duration
    num_rollouts = args.num_rollouts
    warmup_time = args.warmup_time
    train_time = args.train_time

    num_batches = 15

    # observation in.
    if(len(ob_space.shape)==1):
        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[None, ob_space.shape[0]])
    elif(len(ob_space.shape)==2):
        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[None, ob_space.shape[0] * ob_space.shape[1]])
    elif(len(ob_space.shape)==3):
        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[None, ob_space.shape[0] * ob_space.shape[1] * ob_space.shape[2]])
    else:
        raise Exception("unsupported observer space shape (%d)" % len(ob_space.shape))
    # ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[None, 104])

    # features = Features(name="features", ob=ob)
    
    gs_policy = GuessStepsPolicy(name="guess_steps", ob=ob, hid_size=32, num_hid_layers=5)
    policy = Policy(name="policy", ob=ob, ac_space=ac_space, hid_size=32, num_hid_layers=2, num_subpolicies=num_subs)
    old_policy = Policy(name="old_policy", ob=ob, ac_space=ac_space, hid_size=32, num_hid_layers=2, num_subpolicies=num_subs)

    sub_policies = [SubPolicy(name="sub_policy_%i" % x, ob=ob, ac_space=ac_space, hid_size=32, num_hid_layers=2) for x in range(num_subs)]
    old_sub_policies = [SubPolicy(name="old_sub_policy_%i" % x, ob=ob, ac_space=ac_space, hid_size=32, num_hid_layers=2) for x in range(num_subs)]

    learner = Learner(env, policy, old_policy, sub_policies, old_sub_policies, gs_policy, comm, clip_param=0.2, entcoeff=0, optim_epochs=10, optim_stepsize=3e-5, optim_batchsize=64)
    rollout = rollouts.traj_segment_generator(policy, sub_policies, gs_policy, env, macro_duration, num_rollouts, stochastic=True, args=args)

    hasRandomizeCorrect = hasattr(env,"env") and hasattr(env.env,"randomizeCorrect")
    for x in range(100000):
        callback(x)
        if x == 0:
            learner.syncSubpolicies()
            print("synced subpols")
        # Run the inner meta-episode.

        policy.reset()
        learner.syncGuessStepsPolicies()
        learner.syncMasterPolicies()

        if hasRandomizeCorrect:
            env.env.randomizeCorrect()
            shared_goal = comm.bcast(env.env.realgoal, root=0)
            env.env.realgoal = shared_goal
            print("It is iteration %d so i'm changing the goal to %s" % (x, env.env.realgoal))
        mini_ep = 0 if x > 0 else -1 * (rank % 10)*int(warmup_time+train_time / 10)
        # mini_ep = 0

        totalmeans = []
        while mini_ep < warmup_time+train_time:
            mini_ep += 1
            # rollout
            rolls = rollout.__next__()
            
            
            allrolls = []
            allrolls.append(rolls)
            # train theta
            rollouts.add_advantage_macro(rolls, macro_duration, 0.99, 0.98)
            
            
            gmean, lmean = learner.updateMasterPolicy(rolls)
            
            if gmean>0:
                learner.updateGuessStepsPolicyLoss(rolls)
                #print("steps:")
                #print(rolls["steps"])
                #print("gs_vpreds:")
                #print(rolls["gs_vpreds"])
                print("gs mean:")
                print(U.eval(tf.reduce_mean(tf.square(rolls["gs_vpreds"]-rolls["steps"]))))
            
            # train phi
            test_seg = rollouts.prepare_allrolls(allrolls, macro_duration, 0.99, 0.98, num_subpolicies=num_subs)
            learner.updateSubPolicies(test_seg, num_batches, (mini_ep >= warmup_time))

            # log
            print(("%d: global: %s, local: %s" % (mini_ep, gmean, lmean)))
            if args.s:
                totalmeans.append(gmean)
                with open('outfile'+str(x)+'.pickle', 'wb') as fp:
                    pickle.dump(totalmeans, fp)

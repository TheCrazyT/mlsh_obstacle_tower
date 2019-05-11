REM python main.py --task CartPole-v1 --num_subs 2 --macro_duration 1000 --num_rollouts 2000 --warmup_time 20 --train_time 30 --replay False CartPole
set PATH=%PATH%;d:\Program Files\Microsoft MPI\Bin\
set PYTHONPATH=%PYTHONPATH%;d:\Backups_and_projects\sources\mlsh_obstacle_tower
set PYTHONPATH=%PYTHONPATH%;d:\Backups_and_projects\sources\mlsh_obstacle_tower\gym
set PYTHONPATH=%PYTHONPATH%;d:\Backups_and_projects\sources\mlsh_obstacle_tower\rl-algs
..\..\tensorflow_env.bat
cmd
import sys, time
sys.path.append("..")
import argparse
import gymnasium as gym

from envs.reacher_v6 import ReacherEnv
from envs.reacher3_v6 import Reacher3Env
from envs.marrtino_arm import MARRtinoArmEnv

import numpy as np
import pid
import inverse_kinematics as ik

Kp = [3, 3]         # Tune if system is too slow
Ki = [0.05, 0.05]     # Tune if there is steady-state error
Kd = [0.12, 0.12]   # Tune if system oscillates
error_sum, previous_error = np.zeros(2), np.zeros(2)

def print_header(njoints, ndim):
    for i in range(njoints):
        print(f"j{i}; ", end="")
    for i in range(njoints):
        print(f"cos(j{i}); ", end="")
    for i in range(njoints):
        print(f"sin(j{i}); ", end="")
    if ndim==2:
        print("ee_x; ee_y; ee_qw; ee_qz")
    elif ndim==3:
        print("ee_x; ee_y; ee_z; ee_qw; ee_qx; ee_qy; ee_qz")


def print_obs(obs, njoints, ndim):
    # j_i; cosx(j_i); sin(j_i)
    r = ""
    for i in range(0,3*njoints):
        r = r + f"{obs[i]:6.3f}; "
    if ndim==2:
        # fingertip pose  ee_x;ee_y;ee_qw;ee_qz
        for i in range(-6,-2):
            r = r + f"{obs[i]:6.3f}; "
    elif ndim==3:
        # fingertip pose  ee_x;ee_y;ee_z;ee_qw;ee_qx;ee_qy;ee_qz
        for i in range(-10,-3):
            r = r + f"{obs[i]:6.3f}; "
    print(r[0:-2])

def dorun(args):

    render_mode="human" if args.render else None
        
    if args.env=='r2':
        env = ReacherEnv(render_mode=render_mode)
    elif args.env=='r3':
        env = Reacher3Env(render_mode=render_mode)
    elif args.env=='r5':
        env = MARRtinoArmEnv(render_mode=render_mode)
    else:
        print(f"Unknown environment {args.env}")
        sys.exit(1)

    #print(f"Observation: {env.observation_space}")
    #print(f"Action: {env.action_space}")

    if args.log:
        print_header(env.njoints, env.ndim)

    
    observation, info = env.reset(seed=args.seed)
    env.action_space.seed(seed=args.seed)
    if args.log:
        print_obs(observation, env.njoints, env.ndim)

    # Tested poses for videos
    x_target, y_target, theta_target = 0.1, 0.1, 0
    #x_target, y_target, theta_target = 0.1, 0.1, np.pi/2
    #x_target, y_target, theta_target = 0.2, 0.0, 0
    #x_target, y_target, theta_target = 0.0, 0.2, np.pi/2
    #x_target, y_target, theta_target = 0.1, 0.0, np.pi/4
    #x_target, y_target, theta_target = 0.0, 0.0, np.pi
    target_pose = np.array([x_target, y_target, theta_target])
    initial_guess = [0.0, 0.0]
    target_angles = ik.inverse_kinematics_r2(target_pose, initial_guess)
    env.goal = np.array([x_target, y_target])

    tolerance = 0.01
    for _ in range(1,args.steps):
        current_angles = observation[:2]
        dt = 0.01
        action = pid.pid_control(target_angles,  current_angles, dt, error_sum,
                                 previous_error, Kp, Ki, Kd)
        observation, reward, terminated, truncated, info = env.step(action)
        fingertip_position = observation[-6:-4]
        distance_to_goal = np.linalg.norm(fingertip_position - env.goal)
        print(f'Error: {distance_to_goal}')
        if distance_to_goal < tolerance:
            input('Goal reached!')
            break

        if args.log:
            print_obs(observation, env.njoints, env.ndim)
        if render_mode=="human":
            time.sleep(0.1)
        if terminated or truncated:
            observation, info = env.reset()

    env.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-env", type=str, default="r2",
        help="environment [r2,r3,r5] (default: r2)")
    parser.add_argument("-steps", type=int, default=10000,
        help="Execution steps (default: 10,000)")
    parser.add_argument("-seed", type=int, default=1000,
        help="Random seed (default: 1000)")
    parser.add_argument('--render', default = False, action ='store_true',
        help='Enable rendering')
    parser.add_argument('--log', default = False, action ='store_true',
        help='Enable data log')

    args = parser.parse_args()
    dorun(args)




# Controller davidev2
import pystk
import numpy as np


def control(aim_point, current_vel, last_rescue, track, action_num):
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in screen coordinate frame [-1..1]
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """
    action = pystk.Action()

    """
    Your code here
    Hint: Use action.acceleration (0..1) to change the velocity. Try  a target_velocity (e.g. 20).
    Hint: Use action.brake to True/False to brake (optionally)
    Hint: Use action.steer to turn the kart towards the aim_point, clip the steer angle to -1..1
    Hint: You may want to use action.drift=True for wide turns (it will turn faster)
    """    
    
    x = aim_point[0]
    y = aim_point[1]

    # initialize reward
    reward = 0

    # faster the better
    reward += current_vel

    # punish for not aiming at the right direction
    reward -= 10 * aim_point[0]

    if y>-1: angle = np.arctan(x/(y/2+0.5)) / np.pi * 180
    elif x==0: angle = 0
    elif x>=0: angle = 90
    else: angle = -90
    
    "init"
    action.acceleration = 0
    # action.brake = False
    action.steer = 0
    # action.drift = False
    # action.nitro = False

    if action_num == 0:
        action.brake = True
        action.steer = -1
    
    elif action_num == 1:
        action.nitro = True
        action.steer = -1
    
    elif action_num == 2:
        action.drift = True
        action.steer = -1

    elif action_num == 3:
        action.brake = True
        action.steer = 1
    
    elif action_num == 4:
        action.nitro = True
        action.steer = 1
    
    elif action_num == 5:
        action.drift = True
        action.steer = 1

    
    speed_th = 10
    thresh = 30 #turn threshold
    rescue_thresh = 80
    if angle > rescue_thresh:
        action.rescue = True
    if angle < -rescue_thresh:
        action.rescue = True
    elif angle > thresh: #hard turn right
        # action.steer = 1
        # action.drift = True
        if current_vel<speed_th:
            action.acceleration = 1
            # action.drift = False
    # elif angle < -thresh: #hard turn left
    #     action.steer = -1
        # action.drift = True

    else:
        # action.steer = angle/thresh
        action.acceleration = 1
        # if angle < 20 or current_vel < 20: #if slow or straght, boost
        #       action.nitro = True

    # we dont like rescue
    if action.rescue:
        reward -= 100
        # print("rescue on. -10000 rewards")

    # round up to 2nd decimal places, then if 0, reward
    if abs(round(aim_point[0], 2)) == 0.00:
        reward += 100
        # print("good driving. +100 rewards")
    
    # print("reward: ", reward)
    
    return action, reward



if __name__ == '__main__':
    from utils import PyTux
    from argparse import ArgumentParser

    def test_controller(args):
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)
import pystk


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
    Hint: Use action.acceleration (0..1) to change the velocity. Try targeting a target_velocity (e.g. 20).
    Hint: Use action.brake to True/False to brake (optionally)
    Hint: Use action.steer to turn the kart towards the aim_point, clip the steer angle to -1..1
    Hint: You may want to use action.drift=True for wide turns (it will turn faster)
    """
    import numpy as np
    steer_gain = 4.1
    skid_thresh = 0.18
    target_vel = 30

    # initialize reward
    reward = 0

    #this seems to initialize an object
    action = pystk.Action()
    rescue_thresh = 0.9
    global secondlast_rescue
    global stuck_error
    if(last_rescue == 0):
        secondlast_rescue = 0
        stuck_error = False
        stuck_count = 0

    #compute acceleration
    action.acceleration = 1

    if action_num == 0:
        action.brake = True
    
    elif action_num == 1:
        action.nitro = True
    
    elif action_num == 2:
        action.drift = True

    elif action_num == 3:
        action.fire == True
    
    else:
        print("unknown action!!!!!!")
    
    # if current_vel > target_vel:
    # 	action.brake = True
    # 	action.nitro = False
    # else:
    # 	action.brake = False	
    # 	action.nitro = True

    # faster the better
    reward += current_vel

    # punish for not aiming at the right direction
    reward -= 10 * aim_point[0]

    if(stuck_error):
        aim_point[0] = 0
    # Compute steering
    action.steer = np.clip(steer_gain * aim_point[0] * 1.5, -1, 1)
    if abs(aim_point[0]) > rescue_thresh and track != "cocoa_temple":
       action.rescue = True
    # Compute skidding
    # if abs(aim_point[0]) > skid_thresh:
    #     action.drift = True
    # else:
    #     action.drift = False
    if(secondlast_rescue != last_rescue):
        if(last_rescue - secondlast_rescue < 50 and last_rescue - secondlast_rescue > 30):
            stuck_error = True
            secondlast_rescue = last_rescue
        secondlast_rescue = last_rescue

    # we dont like rescue
    if action.rescue:
        reward -= 10000
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
        import numpy as np
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

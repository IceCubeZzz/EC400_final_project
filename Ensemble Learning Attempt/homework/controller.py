import pystk


def control(aim_point, current_vel):
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
    STEERING_MULTIPLIER = 3.2
    DRIFT_ENGAGE_MAGNITUDE = .525
    # horizontal magnitude at which we start slowing down kart
    MAX_HORIZONTAL_MAGNITUDE = .69
    # amount by which we should decrease acceleration once MAX_HORIZONTAL_MAGNITUDE has been reached
    ACCELERATION_DECREMENT_AT_MAX = .5
    # maximum horizontal magnitude for which acceleration is set to 1
    MAX_ACCELERATION_MAGNITUDE = .05

    action.brake = False
    action.nitro = True

    # if target is in left of screen, steer left
    # if aim_point[0] < 0:
    #    action.steer = aim_point[0]
    # if target is in right of screen, steer right
    # elif aim_point[0] > 0:
    #    action.steer = aim_point[0]

    action.steer = aim_point[0] * STEERING_MULTIPLIER

    # check whether should set drift to true
    horizontal_magnitude = abs(aim_point[0])
    if horizontal_magnitude > DRIFT_ENGAGE_MAGNITUDE:
        action.drift = True
    else:
        action.drift = False

    if horizontal_magnitude < MAX_ACCELERATION_MAGNITUDE:
        action.acceleration = 1
    else:
        action.acceleration = .88

    if horizontal_magnitude > MAX_HORIZONTAL_MAGNITUDE:
        action.acceleration = 1 - \
            (horizontal_magnitude * ACCELERATION_DECREMENT_AT_MAX)
        action.brake = True

    # clamp steering value
    action.steer = max(min(action.steer, 1), -1)

    return action


if __name__ == '__main__':
    from utils import PyTux
    from argparse import ArgumentParser

    def test_controller(args):
        import numpy as np
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(
                t, control, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()

    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)

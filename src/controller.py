import pystk
import numpy as np
import torch


# Increases sensitivity to changes in aim point
AIM_POINT_STEERING_SCALAR = 5

# Straight away steering angle threshold for being able to boost
STRAIGHT_AWAY_THRESHOLD = 1.2 

# Target velocity. Increases here require 
# steering sensitivity to increase and 
# tight turn sensitivty decrease
TARGET_VELOCITY = 25

# Sensitivity to Steer angle for drifting
TIGHT_TURN_THRESHOLD = 2.5

# Based on verbose runs, try using a brake when turn exceeds threshold
BRAKING_TURN_THRESHOLD = 2.5

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

    aim_point_x = aim_point[0]
    aim_point_y = aim_point[1]

    # Multiply aim_point_x by scalar to increase sensitivity
    steering_angle = aim_point_x * AIM_POINT_STEERING_SCALAR

    # Clip steering angle to work with pystk.Action.steer range (-1..1)
    action.steer = np.clip(steering_angle, -1, 1)

    # Set velocity
    set_velocity(action, current_vel)

    # drift if this is a tight turn
    if is_tight_turn(steering_angle):
        action.drift = True
    
    # brake if this turn requires braking
    if turn_requires_braking(steering_angle):
        action.brake = True

    # Nitro if this is a straight away
    if is_straight_away(steering_angle):
        action.nitro = True    

    return action

def set_velocity(action, current_velocity):
    if current_velocity < TARGET_VELOCITY:
        action.acceleration = 1

def turn_requires_braking(steering_angle):
    return abs(steering_angle) > BRAKING_TURN_THRESHOLD

def is_tight_turn(steering_angle):
    return abs(steering_angle) > TIGHT_TURN_THRESHOLD

def is_straight_away(steering_angle):
    return abs(steering_angle) < STRAIGHT_AWAY_THRESHOLD

if __name__ == '__main__':
    from .utils import PyTux
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

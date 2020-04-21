from pysc2.agents import base_agent
from pysc2.env import sc2_env, run_loop
from pysc2.lib import actions, features, units
from absl import app
import skfuzzy as fuzz
import numpy as np
import math
from skfuzzy import control as ctrl
import random

HP_OF_STALKER = 160 # use shields ? 80 hp + 80 shield

# ---------------- FUZZY -----------------
distance = ctrl.Antecedent(np.arange(0,85,5), 'distance')
life = ctrl.Antecedent(np.arange(0,HP_OF_STALKER+1,1), 'life')
action = ctrl.Consequent(np.arange(0,3,1), 'action')

action['blink'] = fuzz.trimf(action.universe, [0,0,0])
action['run'] = fuzz.trimf(action.universe, [1,1,1])
action['attack'] = fuzz.trimf(action.universe, [2,2,2])

distance.automf(3)
life.automf(3)

rule1 = ctrl.Rule(life['poor'] & distance['poor'], action['blink'])
rule2 = ctrl.Rule(life['poor'] & distance['average'], action['run'])
rule3 = ctrl.Rule(life['poor'] & distance['good'], action['attack'])
rule4 = ctrl.Rule(life['average'] & distance['poor'], action['blink'])
rule5 = ctrl.Rule(life['average'] & distance['average'], action['run'])
rule6 = ctrl.Rule(life['average'] & distance['good'], action['attack'])
rule7 = ctrl.Rule(life['good'] & distance['poor'], action['attack'])
rule8 = ctrl.Rule(life['good'] & distance['average'], action['attack'])
rule9 = ctrl.Rule(life['good'] & distance['good'], action['attack'])

# TODO : find good thresholds (with smart running)
LIMIT_FOR_ATTACK = 0.9
LIMIT_FOR_RUN = 0.5


def string_action(score):
    if score > LIMIT_FOR_ATTACK : return "attack"
    if score > LIMIT_FOR_RUN : return "run"
    return "blink"

# ---------------- STRACRAFT -----------------
_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

MAX_COORD = 84
MIN_COORD = 0

TOP_LEFT =[MIN_COORD,MIN_COORD]
TOP_RIGHT=[MAX_COORD,MIN_COORD]
BOTTOM_LEFT=[MIN_COORD,MAX_COORD]
BOTTOM_RIGHT=[MAX_COORD,MAX_COORD]
CORNERS = [TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT]

_PLAYER_HOSTILE = 4

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

FUNCTIONS = actions.FUNCTIONS
RAW_FUNCTIONS = actions.RAW_FUNCTIONS

def verify_coord(coord):
    if coord > MAX_COORD :
        return MAX_COORD
    if coord < MIN_COORD:
        return MIN_COORD
    return coord


def _xy_locs(mask):
    """Mask should be a set of bools from comparison with a feature layer."""
    y, x = mask.nonzero()
    return list(zip(x, y))


def select_one_stalker(stalker):
    return actions.FUNCTIONS.select_point("select", (stalker.x, stalker.y))


def get_units_by_type(obs, unit_type):
    return [unit for unit in obs.observation.feature_units
            if unit.unit_type == unit_type]


def can_do(obs, action):
    return action in obs.observation.available_actions


def attack(obs) :
    if can_do(obs, FUNCTIONS.Attack_screen.id):
        player_relative = obs.observation.feature_screen.player_relative
        zealots = _xy_locs(player_relative == _PLAYER_ENEMY)
        if not zealots:
            return FUNCTIONS.no_op()

        # Find the zealot with highest y.
        target = zealots[np.argmax(np.array(zealots)[:, 1])]
        return FUNCTIONS.Attack_screen("now", target)

# TODO : implement "smart" run
#def run(mean_stalker_x,mean_stalker_y, mean_zealot_x, mean_zealot_y) :
    #target_x = verify_coord(2*mean_stalker_x - mean_zealot_x)
    #target_y = verify_coord(2*mean_stalker_y - mean_zealot_y)

    #target = [target_x, target_y]
    #return actions.FUNCTIONS.Move_screen("now", target)


def unit_type_is_selected(obs, unit_type):
    if (len(obs.observation.single_select) > 0 and
            obs.observation.single_select[0].unit_type == unit_type):
        return True

    if (len(obs.observation.multi_select) > 0 and
            obs.observation.multi_select[0].unit_type == unit_type):
        return True

    return False


def select_all_stalkers(stalkers):
    stalker = random.choice(stalkers)
    return actions.FUNCTIONS.select_point("select_all_type", (stalker.x,stalker.y))




class DefeatZealotAgentRunaway(base_agent.BaseAgent):

    def __init__(self):
        super(DefeatZealotAgentRunaway, self).__init__()
        
        self.current_target = None
        self.current_action = None
        self.action_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
        self.action_simulation = ctrl.ControlSystemSimulation(self.action_ctrl)

    def take_decision(self, life, distance):
        self.action_simulation.input['life'] = life
        self.action_simulation.input['distance'] = distance

        self.action_simulation.compute()
        score = self.action_simulation.output['action']


        return string_action(score)
    
    def run(self, stalkerX, stalkerY, zealotX, zealotY):
        # Stalker in top left corner
        if(stalkerX<=(MAX_COORD/4) and stalkerY<=(MAX_COORD/4)):
            print("enter top left")
            available_corners = [BOTTOM_LEFT, TOP_RIGHT]
            distances = [math.sqrt(math.pow(zealotX-corner[0],2) 
                        + math.pow(zealotY-corner[1],2))
                        for corner in available_corners]
            target = available_corners[np.argmax(np.array(distances))]
            self.current_target = target
        # Stalker in top right corner
        elif(stalkerX>=((MAX_COORD/4)*3) and stalkerY<=(MAX_COORD/4)):
            print("enter top right")
            available_corners = [TOP_LEFT, BOTTOM_RIGHT]
            distances = [math.sqrt(math.pow(zealotX-corner[0],2) 
                        + math.pow(zealotY-corner[1],2))
                        for corner in available_corners]
            target = available_corners[np.argmax(np.array(distances))]
            self.current_target = target
        # Stalker in bottom left corner
        elif(stalkerX<=(MAX_COORD/4) and stalkerY>=((MAX_COORD/4)*3)):
            print("enter bottom left")
            available_corners = [TOP_LEFT, BOTTOM_RIGHT]
            distances = [math.sqrt(math.pow(zealotX-corner[0],2) 
                        + math.pow(zealotY-corner[1],2))
                        for corner in available_corners]
            target = available_corners[np.argmax(np.array(distances))]
            self.current_target = target
        # Stalker in bottom right corner
        elif(stalkerX>=((MAX_COORD/4)*3) and stalkerY>=((MAX_COORD/4)*3)):
            print("enter bottom right")
            available_corners = [BOTTOM_LEFT, TOP_RIGHT]
            distances = [math.sqrt(math.pow(zealotX-corner[0],2) 
                        + math.pow(zealotY-corner[1],2))
                        for corner in available_corners]
            target = available_corners[np.argmax(np.array(distances))]
            self.current_target = target
        # Stalker not in a corner
        elif self.current_target is None:
            distances = [math.sqrt(math.pow(stalkerX-corner[0],2) 
                        + math.pow(stalkerY-corner[1],2))
                        for corner in CORNERS]
            # select the two nearest corners
            idx_target1 = np.argmin(np.array(distances))
            copyDistances = list(distances)
            copyCorners = list(CORNERS)
            del copyCorners[idx_target1]
            del copyDistances[idx_target1]
            idx_target2 = np.argmin(np.array(copyDistances))
            # choose the one which is the most far away from the zealots
            target1 = CORNERS[idx_target1]
            target2 = copyCorners[idx_target2]
            targets = [target1,target2]
            zealots_dist = [math.sqrt(math.pow(zealotX-corner[0],2) 
                            + math.pow(zealotY-corner[1],2))
                            for corner in targets]
            target = targets[np.argmax(np.array(zealots_dist))]
            self.current_target = target
        else:
            target = self.current_target
        print(target)
        return actions.FUNCTIONS.Move_screen("now", target)

    def step(self, obs):
        super(DefeatZealotAgentRunaway, self).step(obs)

        stalkers = get_units_by_type(obs, units.Protoss.Stalker)
        # stalker coordinates
        mean_stalker_x = np.mean([s.x for s in stalkers])
        mean_stalker_y = np.mean([s.y for s in stalkers])

        y, x = (obs.observation["feature_screen"][_PLAYER_RELATIVE] == _PLAYER_ENEMY).nonzero()

        mean_zealot_x = x.mean()
        mean_zealot_y = y.mean()

        if self.steps % 20 == 0:
            print("-------------------------")
            avg_health = 0
            stalker = (mean_stalker_x,mean_stalker_y)
            zealot = (mean_zealot_x,mean_zealot_y)
            distance = math.sqrt(math.pow(mean_stalker_y-mean_zealot_y,2) + math.pow(mean_stalker_x-mean_zealot_x,2) )

            # TODO : different weight for hp or shield ???
            # Move each stalker individually ?
            for s in stalkers:
                avg_health += (s.shield + s.health)
            if len(stalkers) != 0:
                avg_health /= len(stalkers)

            self.current_action = self.take_decision(life=avg_health, distance=distance)

        # if stalkers are not selected :
        if not unit_type_is_selected(obs, units.Protoss.Stalker):
            return select_all_stalkers(stalkers)

        if self.current_action == "blink":
            if unit_type_is_selected(obs, units.Protoss.Stalker):
                if can_do(obs, FUNCTIONS.Effect_Blink_screen.id):
                    x = random.randint(0, 83)
                    y = random.randint(0, 83)
                    return FUNCTIONS.Effect_Blink_screen("now", (x, y))
                else :
                    return self.run(mean_stalker_x, mean_stalker_y, mean_zealot_x, mean_zealot_y)

        if self.current_action == "run":
            # Run away from the zealots
            if unit_type_is_selected(obs, units.Protoss.Stalker):
                return self.run(mean_stalker_x, mean_stalker_y, mean_zealot_x, mean_zealot_y)

        if self.current_action == "attack" :
            return attack(obs)

        return FUNCTIONS.no_op()


def main(unused_argv):
    try:
        with sc2_env.SC2Env(
                # Select a map
                map_name="DefeatZealotswithBlink",
                # Add players
                players=[sc2_env.Agent(sc2_env.Race.protoss)],
                # Specify interface
                agent_interface_format=features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(screen=84,
                                                           minimap=64),
                    use_feature_units=True),
                # specify how much action we want to do. 22.4 step per seconds
                step_mul=1,
                game_steps_per_episode=0,
                visualize=False) as env:
            run_loop.run_loop([DefeatZealotAgentRunaway()], env)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app.run(main)
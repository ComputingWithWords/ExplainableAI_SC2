from pysc2.agents import base_agent
from pysc2.env import sc2_env, run_loop
from pysc2.lib import actions, features, units
from absl import app
import skfuzzy as fuzz
import numpy as np
import math
from skfuzzy import control as ctrl
import random
import matplotlib.pyplot as plt

UBUNTU_DEACTIVATE_CHAT = False
PLAY_EPISODES = 3
WATCHING_EPISODES = 2
# ------ WEIGHTS ----------
LIMIT_FOR_ATTACK = 0.69
LIMIT_FOR_RUN = 0.55
ADJUST = False
VISUALIZE = False

STALKER_SHIELD_WEIGHT = 1
STALKER_HEALTH_WEIGHT = 1


# ----------- CONSTANTS FROM WEIGHTS --------------
HP_OF_STALKER = 80 * STALKER_HEALTH_WEIGHT + 80 * STALKER_SHIELD_WEIGHT

MAX_X = MAX_Y = 84
MAX_DISTANCE = math.floor(math.sqrt(MAX_X*MAX_X+MAX_Y*MAX_Y))
# ---------------- FUZZY -----------------
distance = ctrl.Antecedent(np.arange(0,MAX_DISTANCE+1,1), 'distance')
life = ctrl.Antecedent(np.arange(0,HP_OF_STALKER+1,1), 'life')

action = ctrl.Consequent(np.arange(0,3,1), 'action')
action['blink'] = fuzz.trimf(action.universe, [0,0,0])
action['run'] = fuzz.trimf(action.universe, [0,0.25,0.5])
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


def get_weighted_health_value(shield, health):
    return health * STALKER_HEALTH_WEIGHT + shield * STALKER_SHIELD_WEIGHT


def string_action(score, LIMIT_FOR_ATTACK, LIMIT_FOR_RUN):
    if score > LIMIT_FOR_ATTACK : return "attack"
    if score > LIMIT_FOR_RUN : return "run"
    return "blink"

#get the membership value of the fuzzy variable
def mv_life(value):
    inval = value
    memb_list = []
    for term in life.terms:
        mval = np.interp(inval, life.universe, life[term].mf)
        if mval>0:
            memb_list.append([term,mval])

    #value_list = [val[1] for val in memb_list]

    #if 0.5 in value_list:
    result = memb_list
    #else:
     #   result = memb_list[np.argmax(value_list)]
    if(len(result)>1 and result[0][1]<result[1][1]):
        result[0], result[1] = result[1], result[0]
    return result

#get the membership value of the fuzzy variable
def mv_distance(value):
    inval = value
    memb_list = []
    for term in distance.terms:
        mval = np.interp(inval, distance.universe, distance[term].mf)
        if mval>0:
            memb_list.append([term,mval])

    #value_list = [val[1] for val in memb_list]

    #if 0.5 in value_list:
    result = memb_list
    #else:
    #    result = memb_list[np.argmax(value_list)]
    if(len(result)>1 and result[0][1]<result[1][1]):
        result[0], result[1] = result[1], result[0]
    return result

# ---------------- STRACRAFT -----------------
_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

ZEALOT_MAX_HEALTH = 150

MAX_COORD = 84
MIN_COORD = 0

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


def select_all_stalkers(stalkers):
    stalker = random.choice(stalkers)
    return actions.FUNCTIONS.select_point("select_all_type", (stalker.x, stalker.y))


class DefeatZealotAgentFuzzy(base_agent.BaseAgent):

    def __init__(self, env):
        super(DefeatZealotAgentFuzzy, self).__init__()
        self.env = env
        
        self.current_target = [0,0]
        self.previous_action = ""
        self.current_action = ""
        self.action_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
        self.action_simulation = ctrl.ControlSystemSimulation(self.action_ctrl)
        self.adjust = ADJUST
        self.LIMIT_FOR_ATTACK = LIMIT_FOR_ATTACK
        self.LIMIT_FOR_RUN = LIMIT_FOR_RUN

    def unit_type_is_selected(self, obs, unit_type):
        if (len(obs.observation.single_select) > 0 and
                obs.observation.single_select[0].unit_type == unit_type):
            return True

        if (len(obs.observation.multi_select) > 0 and
                obs.observation.multi_select[0].unit_type == unit_type):
            return True

        return False

    def attack(self, obs) :
        if self.can_do(obs, FUNCTIONS.Attack_screen.id):
            zealots = get_units_by_type(obs, units.Protoss.Zealot)
            if len(zealots)<=0:
                return FUNCTIONS.no_op()
            lower_life_zealot = None
            lowest_health = ZEALOT_MAX_HEALTH
            for z in zealots:
                if lower_life_zealot is not None:
                    lowest_health = lower_life_zealot.shield + lower_life_zealot.health
                    zealot_health = z.shield + z.health
                    if zealot_health < lowest_health :
                        lower_life_zealot = z
                else:
                    lower_life_zealot = z

            if lowest_health == ZEALOT_MAX_HEALTH:
                player_relative = obs.observation.feature_screen.player_relative
                zealots = _xy_locs(player_relative == _PLAYER_ENEMY)
                if len(zealots)<= 0:
                    return FUNCTIONS.no_op()

                # Find the zealot with highest y.
                target = zealots[np.argmax(np.array(zealots)[:, 1])]
                return FUNCTIONS.Attack_screen("now", target)

            # Find the zealot with highest y.
            target = (lower_life_zealot.x, lower_life_zealot.y)

            return FUNCTIONS.Attack_screen("now", target)
        else:
            return FUNCTIONS.no_op()

    def can_do(self, obs, action):
        return action in obs.observation.available_actions

    def take_decision(self, life, distance, limit_run, limit_attack):
        self.action_simulation.input['life'] = life
        self.action_simulation.input['distance'] = distance

        self.action_simulation.compute()
        score = self.action_simulation.output['action']

        return string_action(score, LIMIT_FOR_RUN=limit_run, LIMIT_FOR_ATTACK=limit_attack)

    def explainable_msg(self, life, distance):
        life_mval = mv_life(life)
        dist_mval = mv_distance(distance)

        msg_action = self.current_action.capitalize()+", because : "
        
        if len(life_mval)>1:
            msg_life = "life is "
            msg_life += str(life_mval[0][0])+" ("+"{:.0%}".format(life_mval[0][1])+")"
            msg_life +=" and "
            msg_life +=str(life_mval[1][0])+" ("+"{:.0%}".format(life_mval[1][1])+")"
            msg_life +=","
        else:
            msg_life = "life is "
            msg_life +=str(life_mval[0][0])+" ("+"{:.0%}".format(life_mval[0][1])+"),"

        if len(dist_mval)>1:
            msg_dist = "distance is "
            msg_dist +=str(dist_mval[0][0])+" ("+"{:.0%}".format(dist_mval[0][1])+")"
            msg_dist +=" and "
            msg_dist +=str(dist_mval[1][0])+" ("+"{:.0%}".format(dist_mval[1][1])+")."
        else:
            msg_dist = "distance is "
            msg_dist +=str(dist_mval[0][0])+" ("+"{:.0%}".format(dist_mval[0][1])+")."

        

        final_msg = msg_action+msg_life+msg_dist
        print(final_msg)
        listMsgToSend = [msg_action, msg_life, msg_dist,".",".",".",".",".","."]

        return listMsgToSend

    def direction(self, x1, y1, x2, y2):
        weight = 10
        direction = [x1-x2, y1-y2]
        normalizer = 1/math.sqrt(math.pow(x1-x2,2)+math.pow(y1-y2,2))
        norm_direct = [direction[i] * weight * normalizer for i in [0,1]]
        return norm_direct

    def run(self, stalkerX, stalkerY, zealotX, zealotY, obs):
        vect_dir = self.direction(stalkerX, stalkerY, zealotX, zealotY)

        if(stalkerX<(MAX_COORD/4) or stalkerX>(MAX_COORD/4)*3 or stalkerY<(MAX_COORD/4) or stalkerY>45):
            if(zealotY>MAX_COORD/2 and (stalkerY<zealotY and stalkerX<(MAX_COORD/4)*3)):
                target = [stalkerX+((vect_dir[0]-vect_dir[1])/2),stalkerY+((vect_dir[1]+vect_dir[0])/2)]
            else:
                target = [stalkerX+((vect_dir[0]+vect_dir[1])/2),stalkerY+((vect_dir[1]-vect_dir[0])/2)]
        else:
            target = [stalkerX+vect_dir[0], stalkerY+vect_dir[1]]

        target[0] = verify_coord(target[0])
        target[1] = verify_coord(target[1])
        if self.can_do(obs, FUNCTIONS.Move_screen.id):
            return FUNCTIONS.Move_screen("now", target)
        else:
            return FUNCTIONS.no_op()

    def step(self, obs):
        super(DefeatZealotAgentFuzzy, self).step(obs)

        player_relative = obs.observation.feature_screen.player_relative
        stalkers = get_units_by_type(obs, units.Protoss.Stalker)
        if len(stalkers) <= 0:
            return FUNCTIONS.no_op()
        # stalker coordinates
        mean_stalker_x = np.mean([s.x for s in stalkers])
        mean_stalker_y = np.mean([s.y for s in stalkers])

        zealots = _xy_locs(player_relative == _PLAYER_ENEMY)
        if len(zealots)<= 0:
            return FUNCTIONS.no_op()

        mean_zealot_x = np.mean([z[0] for z in zealots])
        mean_zealot_y = np.mean([z[1] for z in zealots])

        if self.steps % 10 == 0:
            avg_health = 0

            distance = math.sqrt(math.pow(mean_stalker_y-mean_zealot_y,2) + math.pow(mean_stalker_x-mean_zealot_x,2) )

            # TODO : Move each stalker individually ?
            for s in stalkers:
                avg_health += get_weighted_health_value(s.shield,s.health)
            if len(stalkers) != 0:
                avg_health /= len(stalkers)

            self.current_action = self.take_decision(life=avg_health, distance=distance, limit_run=self.LIMIT_FOR_RUN, limit_attack=self.LIMIT_FOR_ATTACK)

            if not UBUNTU_DEACTIVATE_CHAT :
                if self.previous_action == "":
                    listMsgs = self.explainable_msg(avg_health, distance)
                    for msg in listMsgs:
                        self.env.send_chat_messages([msg])
                    #action.view(sim=self.action_simulation)
                    #plt.show()
                    input("Press Enter to continue...")
                        
                else:
                    if self.previous_action != self.current_action:
                        listMsgs = self.explainable_msg(avg_health, distance)
                        for msg in listMsgs:
                            self.env.send_chat_messages([msg])
                        #action.view(sim=self.action_simulation)
                        #plt.show()
                        input("Press Enter to continue...")
                            

            self.previous_action = self.current_action

        # if stalkers are not selected :
        if not self.unit_type_is_selected(obs, units.Protoss.Stalker):
            return select_all_stalkers(stalkers)

        if self.current_action == "blink":
            if self.unit_type_is_selected(obs, units.Protoss.Stalker):
                if self.can_do(obs, FUNCTIONS.Effect_Blink_screen.id):
                    x = random.randint(0, 83)
                    y = random.randint(0, 83)
                    return FUNCTIONS.Effect_Blink_screen("now", (x, y))
                else :
                    self.current_action = "run"
                    return self.run(mean_stalker_x, mean_stalker_y, mean_zealot_x, mean_zealot_y, obs)

        if self.current_action == "run":
            # Run away from the zealots
            if self.unit_type_is_selected(obs, units.Protoss.Stalker):
                return self.run(mean_stalker_x, mean_stalker_y, mean_zealot_x, mean_zealot_y, obs)

        if self.current_action == "attack" :
            return self.attack(obs)

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
                game_steps_per_episode=1920,
                visualize=VISUALIZE) as env:

            #run_loop.run_loop([], env, max_episodes=PLAY_EPISODES)
            run_loop.run_loop([DefeatZealotAgentFuzzy(env)], env, max_episodes=WATCHING_EPISODES)
            #run_loop.run_loop([], env, max_episodes=PLAY_EPISODES)

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app.run(main)
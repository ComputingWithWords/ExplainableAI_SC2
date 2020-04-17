from pysc2.agents import base_agent
from pysc2.env import sc2_env, run_loop
from pysc2.lib import actions, features, units
from absl import app

import numpy
import random

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_ENEMY = features.PlayerRelative.ENEMY


_PLAYER_HOSTILE = 4

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

FUNCTIONS = actions.FUNCTIONS
RAW_FUNCTIONS = actions.RAW_FUNCTIONS


def _xy_locs(mask):
    """Mask should be a set of bools from comparison with a feature layer."""
    y, x = mask.nonzero()
    return list(zip(x, y))


class DefeatZealotAgentFuzzy(base_agent.BaseAgent):
    def __init__(self):
        super(DefeatZealotAgentFuzzy, self).__init__()

    def transformDistance(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]

        return [x + x_distance, y + y_distance]

    def transformLocation(self, x, y):
        if not self.base_top_left:
            return [64 - x, 64 - y]

        return [x, y]

    def unit_type_is_selected(self, obs, unit_type):
        if (len(obs.observation.single_select) > 0 and
                obs.observation.single_select[0].unit_type == unit_type):
            return True

        if (len(obs.observation.multi_select) > 0 and
                obs.observation.multi_select[0].unit_type == unit_type):
            return True

        return False

    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units
                if unit.unit_type == unit_type]

    def can_do(self, obs, action):
        return action in obs.observation.available_actions

    def select_one_stalker(self, stalker):
        return actions.FUNCTIONS.select_point("select", (stalker.x, stalker.y))

    def attack(self, obs) :
        if self.can_do(obs, FUNCTIONS.Attack_screen.id):
            player_relative = obs.observation.feature_screen.player_relative
            zealots = _xy_locs(player_relative == _PLAYER_ENEMY)
            if not zealots:
                return FUNCTIONS.no_op()

            # Find the zealot with highest y.
            target = zealots[numpy.argmax(numpy.array(zealots)[:, 1])]
            return FUNCTIONS.Attack_screen("now", target)

    def blink_if_low_health(self, stalkers, obs):
        if len(stalkers) > 0:
            if self.unit_type_is_selected(obs, units.Protoss.Stalker):
                if self.can_do(obs, FUNCTIONS.Effect_Blink_screen.id):
                    bad_health = []
                    for stalker in stalkers:
                        # TODO : change with fuzzy
                        if stalker.health < 60:
                            bad_health.append(stalker)
                    if len(bad_health) > 0:
                        x = random.randint(0, 83)
                        y = random.randint(0, 83)
                        return FUNCTIONS.Effect_Blink_screen("now", (x, y))

    def select_all_stalkers(self, stalkers):
        stalker = random.choice(stalkers)
        return actions.FUNCTIONS.select_point("select_all_type", (stalker.x,stalker.y))

    def step(self, obs):
        super(DefeatZealotAgentFuzzy, self).step(obs)

        # Blink if health is under 60
        stalkers = self.get_units_by_type(obs, units.Protoss.Stalker)

        mean_stalker_x = numpy.mean([s.x for s in stalkers])
        mean_stalker_y = numpy.mean([s.y for s in stalkers])

        # if no stalker are selected at all :
        if not self.unit_type_is_selected(obs, units.Protoss.Stalker):
            return self.select_all_stalkers(stalkers)

        if self.unit_type_is_selected(obs, units.Protoss.Stalker):
            y, x = (obs.observation["feature_screen"][_PLAYER_RELATIVE] == _PLAYER_ENEMY).nonzero()

            mean_zealot_x = x.mean()
            mean_zealot_y = y.mean()

            target_x = 2*mean_stalker_x - mean_zealot_x
            target_y = 2*mean_stalker_y - mean_zealot_y

            if target_x < 0 :
                target_x = 0
            if target_y < 0 :
                target_y = 0

            target = [target_x, target_y]
            print(target)

            return actions.FUNCTIONS.Move_screen("now", target)

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
                visualize=True) as env:
            run_loop.run_loop([DefeatZealotAgentFuzzy()], env)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app.run(main)
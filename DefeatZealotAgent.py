# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 18:05:06 2020

@author: Loïc
"""

from pysc2.agents import base_agent
from pysc2.env import sc2_env, run_loop
from pysc2.lib import actions, features, units
from absl import app

import numpy
import random

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

FUNCTIONS = actions.FUNCTIONS
RAW_FUNCTIONS = actions.RAW_FUNCTIONS

def _xy_locs(mask):
  """Mask should be a set of bools from comparison with a feature layer."""
  y, x = mask.nonzero()
  return list(zip(x, y))

class DefeatZealotAgent(base_agent.BaseAgent):
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

    def step(self, obs):
        super(DefeatZealotAgent, self).step(obs)
        
        #Blink if health is under 60
        stalkers = self.get_units_by_type(obs, units.Protoss.Stalker)
        if len(stalkers) > 0:
            if self.unit_type_is_selected(obs, units.Protoss.Stalker):
                if self.can_do(obs, FUNCTIONS.Effect_Blink_screen.id):
                    bad_health = []
                    for stalker in stalkers:
                        # TODO : change with fuzzy
                        if stalker.health < 60:
                          bad_health.append(stalker)
                    if len(bad_health)>0:
                        x = random.randint(0, 83)
                        y = random.randint(0, 83)
                        return FUNCTIONS.Effect_Blink_screen("now", (x,y))
                
        if self.can_do(obs, FUNCTIONS.Attack_screen.id):
            player_relative = obs.observation.feature_screen.player_relative
            zealots = _xy_locs(player_relative == _PLAYER_ENEMY)
            if not zealots:
                return FUNCTIONS.no_op()
            
            # Find the zealot with highest y.
            target = zealots[numpy.argmax(numpy.array(zealots)[:, 1])]
            return FUNCTIONS.Attack_screen("now", target)
            
        
        if self.can_do(obs, FUNCTIONS.select_army.id):
            return FUNCTIONS.select_army("select")
        
        return FUNCTIONS.no_op()
    
def main(unused_argv):
    try:
        with sc2_env.SC2Env(
                #Select a map
                map_name="DefeatZealotsBlink",
                #Add players
                players=[sc2_env.Agent(sc2_env.Race.protoss)],
                #Specify interface
                agent_interface_format=features.AgentInterfaceFormat(
                        feature_dimensions=features.Dimensions(screen=84,
                                                               minimap=64),
                        use_feature_units=True),
                #specify how much action we want to do. 22.4 step per seconds
                step_mul=2,
                game_steps_per_episode=0,
                visualize=False) as env:
            run_loop.run_loop([DefeatZealotAgent()], env)
    except KeyboardInterrupt:
        pass
    
if __name__ == "__main__":
    app.run(main)
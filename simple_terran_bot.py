# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 17:40:34 2020

@author: Loïc
"""

from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
import random

class TerranAgent(base_agent.BaseAgent):
  def __init__(self):
    super(TerranAgent, self).__init__()
    
    self.attack_coordinates = None

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
    super(TerranAgent, self).step(obs)
    
    if obs.first():
      player_y, player_x = (obs.observation.feature_minimap.player_relative ==
                            features.PlayerRelative.SELF).nonzero()
      xmean = player_x.mean()
      ymean = player_y.mean()
      
      if xmean <= 31 and ymean <= 31:
        self.attack_coordinates = (49, 49)
      else:
        self.attack_coordinates = (12, 16)

    marines = self.get_units_by_type(obs, units.Terran.Marine)
    if len(marines) >= 5:
      if self.unit_type_is_selected(obs, units.Terran.Marine):
        if self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id):
          return actions.FUNCTIONS.Attack_minimap("now",
                                                  self.attack_coordinates)

      if self.can_do(obs, actions.FUNCTIONS.select_army.id):
        return actions.FUNCTIONS.select_army("select")
    
    if self.unit_type_is_selected(obs, units.Terran.SCV):
      free_supply = (obs.observation.player.food_cap -
                     obs.observation.player.food_used)
      supply_depots = self.get_units_by_type(obs, units.Terran.SupplyDepot)
      if free_supply == 0 or len(supply_depots) == 0 :
        if self.can_do(obs, actions.FUNCTIONS.Build_SupplyDepot_screen.id):
            x = random.randint(0, 83)
            y = random.randint(0, 83)
          
            return actions.FUNCTIONS.Build_SupplyDepot_screen("now", (x, y))
    

    barracks = self.get_units_by_type(obs, units.Terran.Barracks)
    if len(barracks) == 0:
      if self.unit_type_is_selected(obs, units.Terran.SCV):
        if self.can_do(obs, actions.FUNCTIONS.Build_Barracks_screen.id):
          x = random.randint(0, 83)
          y = random.randint(0, 83)
          
          return actions.FUNCTIONS.Build_Barracks_screen("now", (x, y))
    
      scvs = self.get_units_by_type(obs, units.Terran.SCV)
      if len(scvs) > 0:
        scv = random.choice(scvs)

        return actions.FUNCTIONS.select_point("select_all_type", (scv.x,
                                                                  scv.y))
    
    if self.unit_type_is_selected(obs, units.Terran.Barracks):
      if self.can_do(obs, actions.FUNCTIONS.Train_Marine_quick.id):
        return actions.FUNCTIONS.Train_Marine_quick("now")
    
    barracks = self.get_units_by_type(obs, units.Terran.Barracks)
    if len(barracks) > 0:
        barrack = random.choice(barracks)
        
        return actions.FUNCTIONS.select_point("select_all_type", (barrack.x,
                                                                  barrack.y))

    
    return actions.FUNCTIONS.no_op()

def main(unused_argv):
  agent = TerranAgent()
  try:
    while True:
      with sc2_env.SC2Env(
          map_name="AbyssalReef",
          players=[sc2_env.Agent(sc2_env.Race.terran),
                   sc2_env.Bot(sc2_env.Race.random,
                               sc2_env.Difficulty.very_easy)],
          agent_interface_format=features.AgentInterfaceFormat(
              feature_dimensions=features.Dimensions(screen=84, minimap=64),
              use_feature_units=True),
          step_mul=1,
          game_steps_per_episode=0,
          visualize=True) as env:
          
        agent.setup(env.observation_spec(), env.action_spec())
        
        timesteps = env.reset()
        agent.reset()
        
        while True:
          step_actions = [agent.step(timesteps[0])]
          if timesteps[0].last():
            break
          timesteps = env.step(step_actions)
      
  except KeyboardInterrupt:
    pass
  
if __name__ == "__main__":
  app.run(main)
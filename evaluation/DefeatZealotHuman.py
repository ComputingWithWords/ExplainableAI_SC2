# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 18:05:06 2020

@author: Loïc
"""

from pysc2.agents import base_agent
from pysc2.env import sc2_env, run_loop
from pysc2.lib import actions, features #, units
from absl import app

FUNCTIONS = actions.FUNCTIONS
PLAY_EPISODES = 3

class DefeatZealotAgent(base_agent.BaseAgent):
    def step(self, obs):
        super(DefeatZealotAgent, self).step(obs)
        
        return FUNCTIONS.no_op()
    
def main(unused_argv):
    try:
        with sc2_env.SC2Env(
                #Select a map
                map_name="DefeatZealotswithBlink",
                #Add players
                players=[sc2_env.Agent(sc2_env.Race.protoss)],
                #Specify interface
                agent_interface_format=features.AgentInterfaceFormat(
                        feature_dimensions=features.Dimensions(screen=84,
                                                               minimap=64),
                        use_feature_units=True),
                #specify how much action we want to do. 22.4 step per seconds
                step_mul=1,
                game_steps_per_episode=1920,
                visualize=False) as env:
            run_loop.run_loop([DefeatZealotAgent()], env, max_episodes=PLAY_EPISODES)
    except KeyboardInterrupt:
        pass
    
if __name__ == "__main__":
    app.run(main)
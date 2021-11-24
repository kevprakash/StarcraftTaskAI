import numpy
import numpy as np
from pysc2.agents import base_agent
from pysc2.lib import actions
import Utility as util
import torch
from pysc2.lib import static_data


# python -m pysc2.bin.agent --map MoveToBeacon --agent TestAgent.TestAgent --feature_screen_size 64,64
# python -m pysc2.bin.agent --map CollectMineralShards --agent TestAgent.TestAgent --feature_screen_size 64,64
# python -m pysc2.bin.agent --map FindAndDefeatZerglings --agent TestAgent.TestAgent --feature_screen_size 64,64
# python -m pysc2.bin.agent --map DefeatRoaches --agent TestAgent.TestAgent --feature_screen_size 64,64
# python -m pysc2.bin.agent --map DefeatZerglingsAndBanelings --agent TestAgent.TestAgent --feature_screen_size 64,64
# python -m pysc2.bin.agent --map CollectMineralsAndGas --agent TestAgent.TestAgent --feature_screen_size 64,64
# python -m pysc2.bin.agent --map BuildMarines --agent TestAgent.TestAgent --feature_screen_size 64,64
class TestAgent(base_agent.BaseAgent):
    def step(self, obs):
        super(TestAgent, self).step(obs)


        function_id = numpy.random.choice(obs.observation.available_actions)
        args = [[numpy.random.randint(0, size) for size in arg.sizes]
                for arg in self.action_spec.functions[function_id].args]
        return actions.FunctionCall(function_id, args)
import numpy
from pysc2.agents import base_agent
from pysc2.lib import actions
from statistics import mean, variance


# python -m pysc2.bin.agent --map MoveToBeacon --agent TestAgent.TestAgent --feature_screen_size 64,64
# python -m pysc2.bin.agent --map CollectMineralShards --agent TestAgent.TestAgent --feature_screen_size 64,64
# python -m pysc2.bin.agent --map FindAndDefeatZerglings --agent TestAgent.TestAgent --feature_screen_size 64,64
# python -m pysc2.bin.agent --map DefeatRoaches --agent TestAgent.TestAgent --feature_screen_size 64,64
# python -m pysc2.bin.agent --map DefeatZerglingsAndBanelings --agent TestAgent.TestAgent --feature_screen_size 64,64
# python -m pysc2.bin.agent --map CollectMineralsAndGas --agent TestAgent.TestAgent --feature_screen_size 64,64
# python -m pysc2.bin.agent --map BuildMarines --agent TestAgent.TestAgent --feature_screen_size 64,64
class TestAgent(base_agent.BaseAgent):
    def __init__(self):
        super().__init__()
        self.prevEp = 1
        self.lastScore = 0
        self.scores = []

    def step(self, obs):
        super(TestAgent, self).step(obs)
        if self.prevEp < self.episodes:
            self.scores.append(self.lastScore)
            self.onNewEpisode()
            self.prevEp = self.episodes
        self.lastScore = float(obs.observation["score_cumulative"][0])

        function_id = numpy.random.choice(obs.observation.available_actions)
        args = [[numpy.random.randint(0, size) for size in arg.sizes]
                for arg in self.action_spec.functions[function_id].args]
        return actions.FunctionCall(function_id, args)

    def onNewEpisode(self):
        if len(self.scores) > 1:
            print("Average score: {:.4f} | Variance: {:.4f}".format(mean(self.scores), variance(self.scores)))
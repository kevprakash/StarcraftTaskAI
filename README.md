# StarcraftTaskAI
Reinforcement Learning AI for doing tasks in Starcraft

## Setup Instructions ##
1. Install [Starcraft II](https://starcraft2.com/en-us/)
2. Create a python environment that has Pytorch and numpy
3. Install into your environment PySC2 through pip:
```
pip install pysc2
```
4. In the python environment, open up a terminal at the root of this repo
5. To run an agent (in this case the CGRU Agent) on one of the tasks, run the appropriate command (each is a different task)
```
python -m pysc2.bin.agent --map MoveToBeacon --agent PolicyAgent.CGRUAgent --feature_screen_size 64,64
python -m pysc2.bin.agent --map CollectMineralShards --agent PolicyAgent.CGRUAgent --feature_screen_size 64,64
python -m pysc2.bin.agent --map FindAndDefeatZerglings --agent PolicyAgent.CGRUAgent --feature_screen_size 64,64
python -m pysc2.bin.agent --map DefeatRoaches --agent PolicyAgent.CGRUAgent --feature_screen_size 64,64
python -m pysc2.bin.agent --map DefeatZerglingsAndBanelings --agent PolicyAgent.CGRUAgent --feature_screen_size 64,64
python -m pysc2.bin.agent --map CollectMineralsAndGas --agent PolicyAgent.CGRUAgent --feature_screen_size 64,64
python -m pysc2.bin.agent --map BuildMarines --agent PolicyAgent.CGRUAgent --feature_screen_size 64,64
```
   - The appropriate commands for each agent are commented above their agent class so they can be copy/pasted from there
   - To reset a saved agent for a task, delete the Model_[Map name].pt file that will be created
6. Refer to the [original PySC2 documentation](https://github.com/deepmind/pysc2) for more detailed information on running agents

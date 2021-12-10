import torch
from torch.nn import functional as F
import numpy as np
from pysc2.agents import base_agent
from pysc2.lib import actions
import Model
import Utility as util
import math
import os
from statistics import mean, variance
from pysc2.lib import static_data


# python -m pysc2.bin.agent --map MoveToBeacon --agent PolicyAgent.UnifiedAgent --feature_screen_size 64,64
# python -m pysc2.bin.agent --map CollectMineralShards --agent PolicyAgent.UnifiedAgent --feature_screen_size 64,64
# python -m pysc2.bin.agent --map FindAndDefeatZerglings --agent PolicyAgent.UnifiedAgent --feature_screen_size 64,64
# python -m pysc2.bin.agent --map DefeatRoaches --agent PolicyAgent.UnifiedAgent --feature_screen_size 64,64
# python -m pysc2.bin.agent --map DefeatZerglingsAndBanelings --agent PolicyAgent.UnifiedAgent --feature_screen_size 64,64
# python -m pysc2.bin.agent --map CollectMineralsAndGas --agent PolicyAgent.UnifiedAgent --feature_screen_size 64,64
# python -m pysc2.bin.agent --map BuildMarines --agent PolicyAgent.UnifiedAgent --feature_screen_size 64,64
class UnifiedAgent(base_agent.BaseAgent):
    def __init__(self):
        super().__init__()
        self.model = Model.UnifiedModel()
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=10 ** (-5))

        self.screens = []
        self.selectedActions = []
        self.selectedArgs = []
        self.scores = []
        self.rewards = []
        self.ep = 1
        self.prevEp = 0
        self.mapName = None

    def step(self, obs):
        super(UnifiedAgent, self).step(obs)

        if self.mapName is None:
            self.mapName = obs.observation['map_name']
            self.load()

        if self.prevEp < self.episodes:
            self.onNewEpisode()
            self.prevEp = self.episodes
            self.ep += 1

        networkInput = self.getNetworkInput(obs)

        _, actionPolicy, argPolicies = self.model(networkInput)
        npActionPolicy = actionPolicy.detach().clone().cpu().numpy()
        selectedAction = util.policyActionSelect(npActionPolicy, obs.observation.available_actions)

        args = []
        for arg in self.action_spec.functions[selectedAction].args:
            argumentPolicy = argPolicies[arg.id]
            npArgPolicy = argumentPolicy.detach().clone().cpu().numpy()
            args.append(util.policyArgSelect(npArgPolicy))

        # self.screens.append(networkInput.detach().cpu().numpy())
        self.screens.append(util.detachAllTensors(self.getNetworkInput(obs)))
        self.selectedActions.append(selectedAction)
        self.selectedArgs.append(args)
        self.scores.append(obs.observation["score_cumulative"][0])
        return actions.FunctionCall(selectedAction, args)

    def onNewEpisode(self):
        self.rewards.append(self.reward)

        # Training
        if len(self.scores) > 1:

            scoreDiff = self.scores[-1] - self.scores[0]
            scoreRate = (scoreDiff + 0.0)/len(self.scores)
            print("Score rate for the episode: " + str(scoreRate))

            self.load()
            # self.scores.append(1000 * (self.rewards[-1] - self.rewards[-2]) + self.scores[-1])
            G = util.calculateScoreDeltas(self.scores)
            index = 0
            valueLossTracker = []
            policyScoreTracker = []
            uncertaintyTracker = []
            for screen in self.screens[:-1]:
                cudaScreen = util.moveAllToCuda(screen)
                valueApprox, actionPolicy, argPolicies = self.model(cudaScreen)
                valueApprox2, _, _ = self.model(util.moveAllToCuda(self.screens[index+1]))
                valueDiff = (G[index] + 0.95 * valueApprox2) - valueApprox
                valueLossTracker.append(valueDiff.item() ** 2)

                actionPickChance = actionPolicy[self.selectedActions[index]]
                entropy = -torch.sum(actionPolicy * torch.log(actionPolicy))
                maxEntropy = torch.log2(torch.prod(torch.tensor(actionPolicy.size())))

                argIndex = 0
                argPickChance = []
                for arg in self.action_spec.functions[self.selectedActions[index]].args:
                    targetArgIndex = tuple(self.selectedArgs[index][argIndex])
                    argPickChance.append(argPolicies[arg.id][targetArgIndex])

                    entropy = entropy - torch.sum(argPolicies[arg.id] * torch.log(argPolicies[arg.id]))
                    maxEntropy = maxEntropy + torch.log2(torch.prod(torch.tensor(argPolicies[arg.id].size())))

                    argIndex += 1

                combinedPickChance = actionPickChance
                for apc in argPickChance:
                    combinedPickChance = combinedPickChance * apc

                policyScore = valueDiff.detach() * combinedPickChance
                policyScoreTracker.append(policyScore.item())
                uncertaintyTracker.append(entropy.item()/maxEntropy.item())
                valueLoss = 0.5 * valueDiff ** 2

                modelLoss = (1 * -policyScore) + (1 * valueLoss) + (0.001 * -entropy)

                self.optimizer.zero_grad()
                modelLoss.backward()
                self.optimizer.step()

                index += 1
                if index == len(self.screens)-1:
                    print("\rTraining Completion: " + str(int(((index + 0.0) / len(self.screens[:-1])) * 100))
                          + '% | Value loss: ' + str(mean(valueLossTracker))
                          + " | Policy Score: " + str(mean(policyScoreTracker))
                          + " | Uncertainty: " + str(mean(uncertaintyTracker)))
                elif index == 1:
                    print("Training Completion: " + str(int(((index + 0.0) / len(self.screens[:-1])) * 100))
                          + '% | Value loss: ' + str(mean(valueLossTracker))
                          + " | Policy Score: " + str(mean(policyScoreTracker))
                          + " | Uncertainty: " + str(mean(uncertaintyTracker)), end='')
                else:
                    print("\rTraining Completion: " + str(int(((index + 0.0) / len(self.screens[:-1])) * 100))
                          + '% | Value loss: ' + str(mean(valueLossTracker))
                          + " | Policy Score: " + str(mean(policyScoreTracker))
                          + " | Uncertainty: " + str(mean(uncertaintyTracker)), end='')

            self.save()

        self.screens = []
        self.selectedActions = []
        self.selectedArgs = []
        self.scores = []
        return

    def getNetworkInput(self, obs):
        featureScreen = torch.tensor(obs.observation["feature_screen"]).float().cuda()
        featureScreen = torch.log2(featureScreen + 1)
        featureScreen = torch.unsqueeze(featureScreen, 0)

        minimap = torch.tensor(obs.observation["feature_minimap"]).float().cuda()
        minimap = torch.log2(minimap + 1)
        minimap = torch.unsqueeze(minimap, 0)

        playerData = torch.tensor(obs.observation['player']).float().cuda()
        playerData = torch.log2(playerData + 1)
        playerData = torch.unsqueeze(torch.unsqueeze(playerData, 1), 2)
        playerData = playerData.expand(-1, 64, 64)
        playerData = torch.unsqueeze(playerData, 0)

        return featureScreen, minimap, playerData

    def save(self):
        actionState = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'ep': self.ep
        }
        torch.save(actionState, 'UnifiedAgent/Model_' + str(self.mapName) + '.pt')
        # print("Model Saved")

    def load(self):
        if os.path.exists('UnifiedAgent/Model_' + str(self.mapName) + '.pt'):
            state = torch.load('UnifiedAgent/Model_' + str(self.mapName) + '.pt')
            self.model.load_state_dict(state['state_dict'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.ep = state['ep']


# python -m pysc2.bin.agent --map MoveToBeacon --agent PolicyAgent.CGRUAgent --feature_screen_size 64,64
# python -m pysc2.bin.agent --map CollectMineralShards --agent PolicyAgent.CGRUAgent --feature_screen_size 64,64
# python -m pysc2.bin.agent --map FindAndDefeatZerglings --agent PolicyAgent.CGRUAgent --feature_screen_size 64,64
# python -m pysc2.bin.agent --map DefeatRoaches --agent PolicyAgent.CGRUAgent --feature_screen_size 64,64
# python -m pysc2.bin.agent --map DefeatZerglingsAndBanelings --agent PolicyAgent.CGRUAgent --feature_screen_size 64,64
# python -m pysc2.bin.agent --map CollectMineralsAndGas --agent PolicyAgent.CGRUAgent --feature_screen_size 64,64
# python -m pysc2.bin.agent --map BuildMarines --agent PolicyAgent.CGRUAgent --feature_screen_size 64,64
class CGRUAgent(base_agent.BaseAgent):
    def __init__(self):
        super().__init__()
        self.model = Model.CGRUPolicyModel()
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=10 ** (-5))

        self.screens = []
        self.minimaps = []
        self.playerData = []
        self.selectedActions = []
        self.selectedArgs = []
        self.scores = []
        self.rewards = []
        self.ep = 1
        self.prevEp = 1
        self.mapName = None

        self.epScore = []

    def step(self, obs):
        super(CGRUAgent, self).step(obs)

        if self.mapName is None:
            self.mapName = obs.observation['map_name']
            self.load()

        if self.prevEp < self.episodes:
            self.onNewEpisode()
            self.prevEp = self.episodes
            self.ep += 1

        screen, minimap, playerData = self.getNetworkInput(obs)
        self.screens.append(screen)
        self.minimaps.append(minimap)
        self.playerData.append(playerData)

        startIndex = max(-len(self.screens), -10)
        screenBuffer = torch.stack(util.moveAllToCuda(self.screens[startIndex:]))
        MMBuffer = torch.stack(util.moveAllToCuda(self.minimaps[startIndex:]))

        _, actionPolicy, argPolicies = self.model([screenBuffer, MMBuffer, self.playerData[-1].cuda()])
        npActionPolicy = actionPolicy.detach().clone().cpu().numpy()
        selectedAction = util.policyActionSelect(npActionPolicy, obs.observation.available_actions)


        args = []
        for arg in self.action_spec.functions[selectedAction].args:
            argumentPolicy = argPolicies[arg.id]
            npArgPolicy = argumentPolicy.detach().clone().cpu().numpy()
            args.append(util.policyArgSelect(npArgPolicy))

        # self.screens.append(networkInput.detach().cpu().numpy())

        self.selectedActions.append(selectedAction)
        self.selectedArgs.append(args)
        self.scores.append(obs.observation["score_cumulative"][0])
        return actions.FunctionCall(selectedAction, args)

    def onNewEpisode(self):
        self.rewards.append(self.reward)

        # Training
        if len(self.scores) > 1:

            self.epScore.append(float(self.scores[-1]))
            # scoreDiff = self.scores[-1] - self.scores[0]
            # scoreRate = (scoreDiff + 0.0)/len(self.scores)
            # print("Score rate for the episode: " + str(scoreRate))

            self.load()
            # self.scores.append(1000 * (self.rewards[-1] - self.rewards[-2]) + self.scores[-1])
            G = util.calculateDiscountedScoreDeltas(self.scores)
            # G = util.calculateScoreDeltas(self.scores)
            valueLossTracker = []
            policyScoreTracker = []
            uncertaintyTracker = []
            for index in range(len(self.screens)):
                startIndex = max(0, index-9)
                screenBuffer = torch.stack(util.moveAllToCuda(self.screens[startIndex:index+1]))
                MMBuffer = torch.stack(util.moveAllToCuda(self.minimaps[startIndex:index+1]))

                # screenBuffer2 = torch.stack(util.moveAllToCuda(self.screens[startIndex+1:index + 2]))
                # MMBuffer2 = torch.stack(util.moveAllToCuda(self.minimaps[startIndex+1:index + 2]))

                valueApprox, actionPolicy, argPolicies = self.model([screenBuffer, MMBuffer, self.playerData[index].cuda()])
                # valueApprox2, _, _ = self.model([screenBuffer2, MMBuffer2, self.playerData[index + 1].cuda()])
                # valueDiff = (G[index] + 0.95 * valueApprox2) - valueApprox
                valueDiff = G[index] - valueApprox
                valueLossTracker.append(valueDiff.item() ** 2)

                actionPickChance = actionPolicy[self.selectedActions[index]]
                entropy = -torch.sum(actionPolicy * torch.log2(actionPolicy))
                maxEntropy = torch.log2(torch.prod(torch.tensor(actionPolicy.size())))

                argIndex = 0
                argPickChance = []
                for arg in self.action_spec.functions[self.selectedActions[index]].args:
                    targetArgIndex = tuple(self.selectedArgs[index][argIndex])
                    argPickChance.append(argPolicies[arg.id][targetArgIndex])

                    entropy = entropy - torch.sum(argPolicies[arg.id] * torch.log2(argPolicies[arg.id]))
                    maxEntropy = maxEntropy + torch.log2(torch.prod(torch.tensor(argPolicies[arg.id].size())))

                    argIndex += 1

                combinedPickChance = actionPickChance
                for apc in argPickChance:
                    combinedPickChance = combinedPickChance * apc

                policyScore = valueDiff.detach() * torch.log10(combinedPickChance)
                policyScoreTracker.append(policyScore.item())
                uncertaintyTracker.append(entropy.item()/maxEntropy.item())
                valueLoss = 0.5 * valueDiff**2

                numArgs = len(self.action_spec.functions[self.selectedActions[index]].args) + 1

                modelLoss = (-1 * policyScore) + (1 * valueLoss) + (0.01 * -entropy)
                # modelLoss = -modelLoss

                self.optimizer.zero_grad()
                modelLoss.backward()
                self.optimizer.step()

                if index == len(self.screens) - 1:
                    print("\rTraining Completion: " + str(int(((index + 0.0) / len(self.screens[:-1])) * 100))
                          + '% | Value loss: ' + str(mean(valueLossTracker))
                          + " | Policy Score: " + str(mean(policyScoreTracker))
                          + " | Uncertainty: " + str(mean(uncertaintyTracker)))
                elif index == 0:
                    print("Training Completion: " + str(int(((index + 0.0) / len(self.screens[:-1])) * 100))
                          + '% | Value loss: ' + str(mean(valueLossTracker))
                          + " | Policy Score: " + str(mean(policyScoreTracker))
                          + " | Uncertainty: " + str(mean(uncertaintyTracker)), end='')
                else:
                    print("\rTraining Completion: " + str(int(((index + 0.0) / len(self.screens[:-1])) * 100))
                          + '% | Value loss: ' + str(mean(valueLossTracker))
                          + " | Policy Score: " + str(mean(policyScoreTracker))
                          + " | Uncertainty: " + str(mean(uncertaintyTracker)), end='')

            self.save()

        self.screens = []
        self.minimaps = []
        self.playerData = []
        self.selectedActions = []
        self.selectedArgs = []
        self.scores = []

        if len(self.epScore) > 1:
            print("Average score: {:.4f} | Variance: {:.4f}".format(mean(self.epScore), variance(self.epScore)))

        return

    def getNetworkInput(self, obs):
        featureScreen = torch.tensor(obs.observation["feature_screen"]).float()
        featureScreen = torch.log2(featureScreen + 1)
        featureScreen = torch.unsqueeze(featureScreen, 0)

        minimap = torch.tensor(obs.observation["feature_minimap"]).float()
        minimap = torch.log2(minimap + 1)
        minimap = torch.unsqueeze(minimap, 0)

        playerData = torch.tensor(obs.observation['player']).float()
        playerData = torch.log2(playerData + 1)
        playerData = torch.unsqueeze(torch.unsqueeze(playerData, 1), 2)
        playerData = playerData.expand(-1, 64, 64)
        playerData = torch.unsqueeze(playerData, 0)

        return featureScreen, minimap, playerData

    def save(self):
        actionState = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'ep': self.ep
        }
        torch.save(actionState, 'CGRUAgent/Model_' + str(self.mapName) + '.pt')
        # print("Model Saved")

    def load(self):
        if os.path.exists('CGRUAgent/Model_' + str(self.mapName) + '.pt'):
            state = torch.load('CGRUAgent/Model_' + str(self.mapName) + '.pt')
            self.model.load_state_dict(state['state_dict'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.ep = state['ep']
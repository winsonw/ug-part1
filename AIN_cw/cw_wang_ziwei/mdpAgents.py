# mdpAgents.py
# parsons/20-nov-2017
#
# Version 1
#
# The starting point for CW2.
#
# Intended to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agent here is was written by Simon Parsons, based on the code in
# pacmanAgents.py

from pacman import Directions
from game import Agent
import api


#
# A class that creates a grid that can be used as a map
#
# The map itself is implemented as a nested list, and the interface
# allows it to be accessed by specifying x, y locations.
#
class Grid:

    # Constructor
    #
    # Note that it creates variables:
    #
    # grid:   an array that has one position for each element in the grid.
    # width:  the width of the grid
    # height: the height of the grid
    #
    # Grid elements are not restricted, so you can place whatever you
    # like at each location. You just have to be careful how you
    # handle the elements when you use them.
    def __init__(self, width, height):
        self.width = width
        self.height = height
        subgrid = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(0)
            subgrid.append(row)

        self.grid = subgrid


    def display(self):
        for i in range(self.height):
            for j in range(self.width):
                # print grid elements with no newline
                print self.grid[i][j],
            # A new line after each line of the grid
            print
            # A line after the grid
        print

    def getValue(self, x, y):
        return self.grid[int(y)][int(x)]

    def getHeight(self):
        return self.height

    def getWidth(self):
        return self.width

    #editted version which store the data as with first index invert
    def setValue(self, x, y, value):
        self.grid[self.height - (int(y) + 1)][int(x)] = value

    #get value exactly at the given position
    def getValueAtExact(self,x,y):
        return self.grid[int(x)][int(y)]

    #set value at the exact given position with given value
    def setValueAtExact(self,x,y,value):
        self.grid[int(x)][int(y)] = value



#By using Modified Policy Iteration to interact with the environemnt
class MDPAgent(Agent):

    # Basic Part start------------------------------------
    def __init__(self):
        print "Starting up MDPAgent!"

    def registerInitialState(self, state):
        #Makin Map
        self.makeMap(state)
        self.addWallsToMap(state)

        #initial setting for the whole game
        self.ghostNum = len(api.ghosts(state))
        self.randomActionP = 0.8
        self.DISTANCEASCLOSE = 2 * self.ghostNum
        self.width = self.map.getWidth()
        self.height = self.map.getHeight()
        self.gamma = 0.9

        #initial setting for the beginning of the game
        self.remainFood = 0
        self.ghostPrediction = []
        self.lastState = None
        self.lastDecidedAction = None

    def final(self, state):
        #resetting variable between games
        self.lastState = None
        self.lastDecidedAction = None
        print "Looks like the game just ended!"

    def getAction(self, state):
        self.initialize(state)
        return self.algorithm(state)

    # initialize Map and data generate between action
    def initialize(self,state):
        self.ghostPrediction = self.predictGhostNextMove(state)
        self.foodNearByMap = self.initialNeardbyFoodMap()
        self.updateMap(state)
        self.remainFood = len(api.food(state)) + len(api.capsules(state))
        self.findNearbyFood()
        self.initialRewardMap()

    #initilize the map which going to store the nearby food position for every position
    def initialNeardbyFoodMap(self):
        foodNearByMap = []
        for i in range(self.height):
            foodNearByRow = []
            for j in range(self.width):
                foodNearByRow.append(set())
            foodNearByMap.append(foodNearByRow)
        return foodNearByMap

    #initilize the map which going to store the policy for every position
    def initialPolicyMap(self):
        policyMap = []
        for i in range(self.height):
            policyRow = []
            for j in range(self.width):
                policyRow.append(None)
            policyMap.append(policyRow)
        return policyMap

    #initilize the map which going to store the reward value for every position
    def initialRewardMap(self):
        self.rewardMap = []
        for i in range(self.height):
            rewardRow = []
            for j in range(self.width):
                rewardRow.append(None)
            self.rewardMap.append(rewardRow)

    #initilize the map which going to store the utility value for every position
    def initialUtilMap(self):
        utilMap = []
        for i in range(self.height):
            utilRow = []
            for j in range(self.width):
                if self.map.getValueAtExact(i, j) != '%':
                    utilRow.append(0)
                else:
                    utilRow.append(-1)
            utilMap.append(utilRow)
        return utilMap
    #Basic Part end------------------------------------------


    #Map Making start-------------------------------------------
    def makeMap(self,state):
        corners = api.corners(state)
        height = self.getLayoutHeight(corners)
        width  = self.getLayoutWidth(corners)
        self.map = Grid(width, height)

    def getLayoutHeight(self, corners):
        height = -1
        for i in range(len(corners)):
            if corners[i][1] > height:
                height = corners[i][1]
        return height + 1

    def getLayoutWidth(self, corners):
        width = -1
        for i in range(len(corners)):
            if corners[i][0] > width:
                width = corners[i][0]
        return width + 1

    def addWallsToMap(self, state):
        walls = api.walls(state)
        for i in range(len(walls)):
            self.map.setValue(walls[i][0], walls[i][1], '%')

    #between action, re-painting the map with Pacman, food, ghost, capsule, and the prediction of ghost.
    def updateMap(self, state):
        for i in range(self.map.getHeight()):
            for j in range(self.map.getWidth()):
                if self.map.getValueAtExact(i, j) != '%':
                    self.map.setValueAtExact(i, j, ' ')

        food = api.food(state)
        capsule = api.capsules(state)
        ghosts = api.ghostStatesWithTimes(state)
        pacman = api.whereAmI(state)
        self.map.setValue(pacman[0], pacman[1], 'P')
        for i in range(len(capsule)):
            self.map.setValue(capsule[i][0], capsule[i][1], '1')
            pos = self.processPos(capsule[i])
            self.foodNearByMap[pos[0]][pos[1]].add(pos)
        for i in range(len(food)):
            self.map.setValue(food[i][0], food[i][1], '*')
            pos = self.processPos(food[i])
            self.foodNearByMap[pos[0]][pos[1]].add(pos)
        for i in range(len(ghosts)):
            if ghosts[i][1] <= 3:
                self.map.setValue(ghosts[i][0][0], ghosts[i][0][1], 'G')
                for j in range(len(self.ghostPrediction[i])):
                    if self.map.getValueAtExact(self.ghostPrediction[i][j][0], self.ghostPrediction[i][j][1]) != "G":
                        self.map.setValueAtExact(self.ghostPrediction[i][j][0], self.ghostPrediction[i][j][1], 'M')
    #Map making end---------------------------------------------------------


    #Utils start-----------------------------------------------------------
    #Return the position with the given position going to the given direction
    def positionAfterMove(self,pos,dir):
        if dir == Directions.NORTH: newPos = (pos[0] - 1, pos[1])
        elif dir == Directions.SOUTH: newPos = (pos[0] + 1, pos[1])
        elif dir == Directions.EAST: newPos = (pos[0], pos[1] + 1)
        elif dir == Directions.WEST: newPos = (pos[0], pos[1] - 1)
        else: newPos = pos
        return newPos

    def convertNumDir(self,num):
        if num == 4: return Directions.STOP
        if num == 0: return Directions.NORTH
        if num == 1: return Directions.SOUTH
        if num == 2: return Directions.WEST
        if num == 3: return Directions.EAST

    #return the surround postion which is not wall, contain_stop = True when the given position is counted as well
    def getNearbyPos(self,pos,contain_stop = True):
        nearbyPos =[]
        if contain_stop:
            r = 5
        else:
            r = 4
        for n in range(r):
            dir = self.convertNumDir(n)
            if not self.isFacingWall(pos,dir):
                posAfter = self.positionAfterMove(self.posTurnInt(pos),dir)
                nearbyPos.append(posAfter)
        return nearbyPos

    def posTurnInt(self,pos):
        return (int(pos[0]),int(pos[1]))

    #turn the position List from api into the position that we can understand
    def processPoss(self, objects):
        for i in range(len(objects)):
            objects[i] = self.processPos(objects[i])
        return objects

    #turn the position from api into the position that we can understand
    def processPos(self, pos):
        return (self.height - (pos[1] + 1), pos[0])

    def isFacingWall(self,pos,dir):
        i,j = self.positionAfterMove(pos,dir)
        return self.map.getValueAtExact(i,j) == "%"

    #From 2 given position to decide which direction the previous object went
    def getDir(self,oldPos,newPos):
        if oldPos[1] > newPos[1]:
            return Directions.WEST
        if oldPos[1] < newPos[1]:
            return Directions.EAST
        if oldPos[0] > newPos[0]:
            return Directions.NORTH
        if oldPos[0] < newPos[0]:
            return Directions.SOUTH
        return Directions.STOP

    #for testing propose only: to print out Maps
    def printMap(self, object):
        for i in object:
            for j in i:
                if j is None:
                    print "XX",
                elif isinstance(j,str):
                    print j[:2],
                elif isinstance(j,set):
                    print len(j),
                else:
                    print (int(j))//10,
            print
        print

    #for testing propose only: to check if the last action is random.
    def isRandomAction(self,state):
        if self.lastState == None:
            return None
        actualAction = self.getDir(self.processPos(api.whereAmI(self.lastState)),self.processPos(api.whereAmI(state)))
        return actualAction == self.lastDecidedAction
    #Utils end-------------------------------------------------------------


    #Reward function start---------------------------------------------------------
    def getRewards(self, state, pos):
        if self.rewardMap[pos[0]][pos[1]] != None:
            return self.rewardMap[pos[0]][pos[1]]
        if self.map.getValueAtExact(pos[0],pos[1]) == "G":
            return -990
        if self.map.getValueAtExact(pos[0],pos[1]) == "M":
            return -891
        reward = self.rewardFunction(pos)
        self.rewardMap[pos[0]][pos[1]] = reward
        return reward

    def rewardFunction(self,pos):
        food_value = self.foodValue(pos)
        return 10*food_value

    # find food nearby in the given distance(range)
    def findNearbyFood(self):
        for n in range(self.DISTANCEASCLOSE - 1):
            newFoodMap = self.initialNeardbyFoodMap()
            for i in range(self.height):
                for j in range(self.width):
                    if self.map.getValueAtExact(i,j) != "%":
                        nearyByBlock = self.getNearbyPos((i, j))
                        for pos in nearyByBlock:
                            for x in self.foodNearByMap[pos[0]][pos[1]]:
                                newFoodMap[i][j].add(x)
            self.foodNearByMap = newFoodMap

    #by using the last state to predict the ghost position in the coming round
    def predictGhostNextMove(self,state):
        ghostsPossibleMovement = []
        ghostsPos = self.processPoss(api.ghosts(state))
        if not self.lastState is None:
            lastGhostPos = self.processPoss(api.ghosts(self.lastState))
        for i in range(self.ghostNum):
            ghostPos = ghostsPos[i]
            nearbyPos = self.getNearbyPos(ghostPos,contain_stop=False)
            if len(nearbyPos) >= 2 and not self.lastState is None and nearbyPos.__contains__(lastGhostPos[i]):
                index = nearbyPos.index(lastGhostPos[i])
                nearbyPos.pop(index)
            ghostPossibleMovement = nearbyPos
            ghostsPossibleMovement.append(ghostPossibleMovement)
        return ghostsPossibleMovement

    #use the nearybyfood value plus if the given position is food or not
    def foodValue(self,pos):
        nearbyFood = len(self.foodNearByMap[pos[0]][pos[1]])
        if self.map.getValueAtExact(pos[0],pos[1]) == "*" or self.map.getValueAtExact(pos[0],pos[1]) == "1":
            value = 0.5 + float(nearbyFood-1) / self.remainFood * 0.5
        else:
            value = float(nearbyFood) / self.remainFood * 0.5
        return value
    #Reward function end------------------------------------------------------------

    #Modified Policy Iteration start------------------------------------------------------
    #instead of reach coverage, i decide to just run it 20 times each actions to run more data set
    def algorithm(self,state):

        policyMap = self.initialPolicyMap()
        utilMap = self.initialUtilMap()
        actions = api.legalActions(state)
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)
        for n in range(20):
            utilMap = self.policyEvaluation(state,utilMap,policyMap)
            policyMap = self.policyImprovement(utilMap,policyMap)

        # self.printMap(utilMap)
        # self.printMap(policyMap)
        # self.map.display()

        i,j = self.processPos(api.whereAmI(state))
        chosenActionDir = policyMap[i][j]
        self.lastState = state
        self.lastDecidedAction = chosenActionDir
        return api.makeMove(chosenActionDir, actions)

    def policyEvaluation(self,state, utilMap, policyMap):
        newUtilMap = self.initialUtilMap()
        for i in range(self.height-2):
            for j in range(self.width-2):
                pos = (i+1,j+1)
                if utilMap[i+1][j+1] != -1:
                    reward = self.getRewards(state,pos)
                    newUtilMap[i + 1][j + 1] = reward
                    if policyMap[i+1][j+1] != None:
                        action_i, action_j = self.positionAfterMove((i+1,j+1),policyMap[i+1][j+1])
                        expect = self.randomActionP * utilMap[action_i][action_j] + (1 - self.randomActionP) * self.randomNearbyAction(utilMap,pos)
                        newUtilMap[i+1][j+1] += self.gamma * expect
        return newUtilMap

    def policyImprovement(self,utilMap,policyMap):
        for i in range(self.height - 2):
            for j in range(self.width - 2):
                if utilMap[i + 1][j + 1] != -1:
                    policyMap[i + 1][j + 1] = self.selectLagrestNearby(utilMap,(i + 1, j + 1))
        return policyMap

    #by comparing the utility map to choose the most-rewarded nearby position
    def selectLagrestNearby(self,utilMap,pos):
        max_i = None
        max_j = None
        max_dir = Directions.STOP
        for new_pos in self.getNearbyPos(pos):
            (new_i, new_j) = new_pos
            if (max_i is None) or (utilMap[new_i][new_j] >= utilMap[max_i][max_j]):
                max_i = new_i
                max_j = new_j
                max_dir = self.getDir(pos,new_pos)
        return max_dir

    #randomly choose neaby postion by calculating the expacted value
    def randomNearbyAction(self,utilMap,pos):
        count = 0
        value = 0
        for new_pos in self.getNearbyPos(pos):
            (i, j) = new_pos
            if (utilMap[i][j] != -1):
                value += utilMap[i][j]
                count += 1
        return value / count
    #Modified Policy Iteration end--------------------------------------------------------

















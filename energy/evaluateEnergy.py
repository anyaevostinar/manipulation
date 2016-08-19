# USE THIS PROGRAM TO OBSERVE THE BEHAVIOR OF AN EVOLVED NEAT NETWORK
# USAGE: python evaluateEnergy.py chromo_file log_file

from pyrobot.engine import Engine
from pyrobot.system.config import *
import time
import os
import sys
import math, random
import cPickle as pickle
from neat import config, population, chromosome, genome, visualize
from neat.nn import nn_pure as nn

def eval_fitness(population):
    """
    Evaluate the fitness of the given population.
    """
    for chromo in population:
        chromo.fitness = eval_individual(chromo)
        print "Fitness:", chromo.fitness, "\n"

def eval_individual(chromo, logFP=None):
    """
    Evaluate the fitness of the given individual in the population.
    Each individual encodes the weights and topology of a neural
    network.  Create this network and use it to control a simulated
    robot in the given task.

    In this experiment, the task is to stay alive for as long as
    possible by maintaining the robot's energy level.  Every time step
    the robot's energy level diminishes. It can replenish its energy
    level by eating food. Food is represented as lights.  The network
    has three inputs: two light sensors and one energy warning sensor.
    When the robot's energy level gets below 10, the warning sensor
    begins to register.  It has two motor outputs to control the speed
    of the left and right wheels.

    Fitness value is the number of steps it survived divided by maximum
    possible steps.
    """
    brain = nn.create_ffphenotype(chromo)
    stepCost = 0.3
    initEnergy = 20
    foodEnergy = 10
    steps = 0
    numTrials = 3
    foodEaten = 0
    for trial in range(numTrials):
        energy = initEnergy
        # randomly position robot
        adaptRobot.stop()
        heading = random.random() * math.pi * 2 #radians
        x = random.random() * 5 + 1
        y = random.random() * 5 + 1
        adaptRobot.simulation[0].setPose('adapter', x, y, heading)
        # randomly position lights
        adaptRobot.simulation[0].eval("self.resetLights(0.3,7,7)")
        while energy > 0:
            steps += 1
            energy -= stepCost  # decrease energy for every time step
            brain.flush()
            adaptRobot.update() # update robot's sensor values
            # increase energy when food is eaten
            result = adaptRobot.eat(-1)
            if result > 0:
                foodEaten += 1
                energy += foodEnergy
                if logFP: logFP.write("Ate food\n")
            if energy < foodEnergy:
                warning = 1.0 - energy/float(foodEnergy)
            else:
                warning = 0.0
            ins = adaptRobot.light[0].value + [warning]
            outs = brain.sactivate(ins)
            adaptRobot.motors(outs[0], outs[1])
            if logFP:
                logFP.write("Step %4d Energy %5.2f SENSORS " % (steps, energy))
                logFP.write("Lights %.2f %.2f Warn %.2f Motors %.2f %.2f\n"% \
                            (ins[0], ins[1], ins[2], outs[0], outs[1]))
            if adaptRobot.stall:
                # to speed up simulation, when stalled just add in steps remaining
                steps += math.ceil(energy/stepCost)
                print "ending trial, robot stalled"
                if logFP: logFP.write("Robot stalled\n")
                break
        if logFP: logFP.write("Trial ended\n")

    adaptRobot.stop()
    print "survived for %d steps, ate %d energy pellets" % (steps, foodEaten)
    if logFP: logFP.write("Survived %d steps, ate %d food\n" % (steps, foodEaten))
    maxEnergy = initEnergy + 10*foodEnergy*numTrials
    maxSteps = maxEnergy/stepCost
    return steps/maxSteps

# start up the simulation
pyroConfig = Configuration()
pyroConfig.put("pyrobot", "gui", "tk")
adapter = Engine(robotfile="PyrobotRobot60001.py",
                 simfile="PyrobotSimulator",
                 config=pyroConfig,
                 worldfile="LightAsFood.py")
adaptRobot = adapter.robot

# set up neat
config.load('energy_config')
chromosome.node_gene_type = genome.NodeGene

def main(argv = None):
    """
    The configuration file, energy_config, provides all the essential
    parameters.
    """
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) != 2:
        print "Usage: python evaluateEnergy.py chromo_file log_file"
        return
    else:
        chromo_file = argv[0]
        log_file = argv[1]
    logFP = open(log_file, "w")
    fp = open(chromo_file, "r")
    chromo = pickle.load(fp)
    fp.close()
    print chromo
    visualize.draw_net(chromo, "_"+chromo_file)
    
    result = eval_individual(chromo, logFP)
    print "Fitness:", result
    logFP.write("Fitness %f" % result)
    logFP.close()

if __name__ == '__main__':
    main()


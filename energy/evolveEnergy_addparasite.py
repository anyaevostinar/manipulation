# USE THIS PROGRAM TO RUN FAST EVOLUTION (SIMULATOR IS NOT DISPLAYED)
# USAGE: python evolveEnergy.py

from pyrobot.simulators.pysim import *
from pyrobot.robot.symbolic import Simbot
import time
import os
import sys
import math, random
import cPickle as pickle
from neat import config, population, host_chromosome, genome, visualize
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
    
    
    PARASITES: Parasites have now been introduced and they are the ones evolving.
    The host is held constant and the parasite is evaluated based on how close its host is
    to the position 0, 0 by the time the host runs out of energy (that could complicate things, try making it a set time).
    """
    brain = nn.create_ffphenotype(chromo)

    parasite = nn.create_phenotype(chromo.parasite_chrom)

    stepCost = 0.3
    initEnergy = 20
    foodEnergy = 10
    steps = 0
    numTrials = 3
    foodEaten = 0
    total_distance = 0
    points = 0
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
          
            parasite_manip = parasite.sactivate([0] * brain.num_inputs)

            #make a dictionary of manipulation values to make things easier
            parasite_dict = {}
            val = 0
            while (val < (len(parasite_manip) -1)):
              if (val % 2) == 0:
                #for every even index (except last if odd number of values) put that value as the key and the next value as the value
                #because the parasite outputs a number 0-1, must transform it to a node index
                node_value = int(parasite_manip[val] * brain.num_neurons)
                parasite_dict[node_value] = parasite_manip[val+1]
              val += 1
            
            
            outs = brain.sactivate_parasite(parasite_dict, ins)

            adaptRobot.motors(outs[0], outs[1])
            if logFP:
                logFP.write("Step %4d Energy %5.2f SENSORS " % (steps, energy))
                logFP.write("Lights %.2f %.2f Warn %.2f Motors %.2f %.2f\n"% \
                            (ins[0], ins[1], ins[2], outs[0], outs[1]))
            if adaptRobot.stall:
                # to speed up simulation
                # when stalled just add in steps remaining
                steps += math.ceil(energy/stepCost)
                print "ending trial, robot stalled"
                if logFP: logFP.write("Robot stalled\n")
                break
            sim.step(run=0) # update the simulator directly
        if logFP: logFP.write("Trial ended\n")
        

        #Find out where the host currently is
        coordinates = adaptRobot.simulation[0].getPose(0)
        print coordinates
        if coordinates[0] == 0 and coordinates[1] == 0:
          #if parasite got host to 0, 0 it gets rewarded for how fast it did so
          points += 50 - steps
        else:
          #if parasite didn't get host to 0, 0 it gets rewarded for how close it got
          points += (50 - (coordinates[0] + coordinates[1]))


          #TODO: change the tournament selection to work based on parasite fitness and spread parasite horizontally

    adaptRobot.stop()
    print "survived for %d steps, ate %d energy pellets" % \
          (steps, foodEaten)
    if logFP: logFP.write("Survived %d steps, ate %d food\n" % \
                          (steps, foodEaten))




    return points

# instantiate the simulator directly and set run to 0
sim = TkSimulator((441,434), (22,420), 40.357554, run=0)  
sim.addBox(0, 0, 7, 7)
sim.addLight(3, 6, 0.3)
sim.addLight(4, 6, 0.3)
sim.addLight(2, 4.5, 0.3)
sim.addLight(5, 4.5, 0.3)
sim.addLight(1, 3.5, 0.3)
sim.addLight(6, 3.5, 0.3)
sim.addLight(2, 2.5, 0.3)
sim.addLight(5, 2.5, 0.3)
sim.addLight(3, 1, 0.3)
sim.addLight(4, 1, 0.3)
sim.addRobot(60000, TkPioneer("adapter",3.5, 3.5, 0,
                              ((.225, .225, -.225, -.225),
                               (.175, -.175, -.175, .175))))
sim.robots[0].addDevice(PioneerFrontLightSensors())

# create a symbolic robot
adaptRobot = Simbot(sim, ["localhost", 60000], 0)

# set up neat
config.load('energy_config')
host_chromosome.node_gene_type = genome.NodeGene

def main():
    """
    The configuration file, energy_config, provides all the essential
    parameters.
    """
    # Start the evolutionary process
    population.Population.evaluate = eval_fitness

    pop = population.Population("best_chromo_10")
    start = time.time()
    generations = 11
    ##TODO: figure out where selection is happening and change it to select based on parasite instead of host
    pop.epoch(generations, report=True, save_best=True, \
              checkpoint_interval=None, checkpoint_generation=5)
    stop = time.time()
    elapsed = stop - start
    print "Total time to evolve:", elapsed
    print "Time per generation:", elapsed/float(generations)
    # Plots the evolution of the best/average fitness 
    visualize.plot_stats(pop.stats)
    # Visualizes speciation
    visualize.plot_species(pop.species_log)

if __name__ == '__main__':
    main()


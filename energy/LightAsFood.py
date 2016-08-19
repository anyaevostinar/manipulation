"""
A PyrobotSimulator world. A large room with one robot and
many lights.

(c) 2005, PyroRobotics.org. Licensed under the GNU GPL.
"""

from pyrobot.simulators.pysim import TkSimulator, TkPioneer, \
     PioneerFrontLightSensors

def INIT():
    # (width, height), (offset x, offset y), scale:
    sim = TkSimulator((441,434), (22,420), 40.357554)  
    # x1, y1, x2, y2 in meters:
    sim.addBox(0, 0, 7, 7)
    # (x, y) meters, brightness usually 1 (1 meter radius):
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
    # port, name, x, y, th, bounding Xs, bounding Ys, color
    # (optional TK color name):
    sim.addRobot(60001, TkPioneer("adapter",
                                  3.5, 3.5, 0,
                                  ((.225, .225, -.225, -.225),
                                   (.175, -.175, -.175, .175))))
    # add some sensors:
    sim.robots[0].addDevice(PioneerFrontLightSensors())
    return sim

import matplotlib.pyplot as plt
import numpy as np

TIMESTEP_ms = 1

class PhysThing:
    mass_kg = 0
    v_x_mps = 0 # these x, y, z, are referencing world frame
    v_y_mps = 0
    v_z_mps = 0 # z is up 
    def vel():
        return np.array([self.v_x_mps, self.v_y_mps, self.v_z_mps])

def get_gravity_force(PhysThing):
    PhysThing
    return 

def apply_dynamics():
    get_gravity_force()
    

print(TIMESTEP_ms)
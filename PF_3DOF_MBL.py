import numpy as np
import scipy
from PFMBLocalization import PFMBL
from DifferentialDriveSimulatedRobot import *
from Pose3D import *

class PF_3DOF_MBL(PFMBL):
    def __init__(self, *args):

        zf_dim = 2 # dimensionality of a Cartesian feature observation
        super().__init__(zf_dim, *args)
        
        self.dt = 0.1  # dt is the sampling time at which we iterate the DR
        self.wheelRadius = 0.1  # wheel radius
        self.wheelBase = 0.5  # wheel base
        self.robot.pulse_x_wheelTurns = 4096  # number of pulses per wheel turn

    def GetInput(self):
        """
        Get the input for the motion model.

        :return: * **uk, Qk**. uk: input vector (:math:`u_k={}^B[\Delta x~\Delta y]^T`), Qk: covariance of the input noise
        """

        # **To be completed by the student**.
        """
        Get the input for the motion model. In this case, the input is the readings from both wheel encoders.

        :return: uk:  input vector (:math:`u_k=[n_L~n_R]^T`)
        """

        # TODO: to be completed by the student
        # Get output of encoder via ReadEncoder() function
        uk_pulse, _     = self.robot.ReadEncoders()
        
        # Compute travel distance of 2 wheels [meter] from output of the encoder
        d_L     = uk_pulse[0, 0] * (2*np.pi*self.wheelRadius/self.robot.pulse_x_wheelTurns)
        d_R     = uk_pulse[1, 0] * (2*np.pi*self.wheelRadius/self.robot.pulse_x_wheelTurns)

        # Compute travel distance of the center point of robot between k-1 and k
        d       = (d_L + d_R) / 2.
        # Compute rotated angle of robot around the center point between k-1 and k
        delta_theta_k   = np.arctan2(d_R - d_L, self.wheelBase)

        # Compute xk from xk_1 and the travel distance and rotated angle. Got the equations from chapter 1.4.1: Odometry 
        uk              = np.array([[d],
                                    [0],
                                    [delta_theta_k]])
        
        Rsk = np.diag(np.array([0.5 ** 2, 0.5 ** 2, np.deg2rad(5) ** 2]))  # covariance of simulated displacement noise

        return uk, Rsk 
    
    def GetMeasurements(self):
        """
        Read the measurements from the robot. Returns a vector of range distances to the map features.
        Only those features that are within the :attr:`SimulatedRobot.SimulatedRobot.Distance_max_range` of the sensor are returned.
        The measurements arribe at a frequency defined in the :attr:`SimulatedRobot.SimulatedRobot.Distance_feature_reading_frequency` attribute.

        :return: vector of distances to the map features, covariance of the measurement noise
        """
        
        # **To be completed by the student**.
        range, R_ranges = self.robot.ReadRanges()

        return range, R_ranges

    
    def MotionModel(self, u, noise):
        # **To be completed by the student**.
        # create array of n_particles particle Rsks distributed randomly around u with covariance Q
        # Get number of particles
        n_particles = len(self.particles)

        # Init particle u
        particles_u = np.zeros((n_particles,3))
        for i in range(n_particles):
            # Add noise to the u particle
            particles_u[i,0:3] = u.T + np.random.normal(0.0, np.diag(noise))
            # Motion model
            self.particles[i,0:3] = Pose3D.oplus(self.particles[i,0:3].reshape((3,1)), particles_u[i,0:3].reshape((3,1))).reshape((1,3))
    

if __name__ == '__main__':

    M = [np.array([[-40, 5]]).T,
           np.array([[-5, 40]]).T,
           np.array([[-5, 25]]).T,
           np.array([[-3, 50]]).T,
           np.array([[-20, 3]]).T,
           np.array([[40,-40]]).T]  # feature map. Position of 2 point features in the world frame.

    #Simulation:
    xs0 = np.zeros((6, 1))
    kSteps = 5000
    index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None), IndexStruct("yaw", 2, 0)]
    robot = DifferentialDriveSimulatedRobot(xs0, M)  # instantiate the simulated robot object
    
    # Particle Filter
    x0 = Pose3D(np.zeros((3,1)))  # initial guess
    P0 = np.diag([2**2, 2**2, np.deg2rad(20)**2]) # Initial uncertainty
    n_particles = 50

    #create array of n_particles particles distributed randomly around x0 with covariance P
    particles = np.zeros((n_particles,3))
    for i in range(n_particles):
        particles[i,:] = x0.T + np.random.normal(0.0, np.diag(P0))
    #
    # **To be completed by the student**.
    #
    
    usk=np.array([[0.5, 0.03]]).T
    pf = PF_3DOF_MBL(M, index, kSteps, robot, particles)
    pf.LocalizationLoop(x0, usk)

import numpy as np
import math
from MCLocalization import MCLocalization
import random as rnd
import time

class PFMBL(MCLocalization):
    """
    Particle Filter Map Based Localization class.

    This class defines a Map Based Localization using a Particle Filter. It inherits from :class:`MCLocalization`, so the Prediction step is already implemented.
    It needs to implement the Update function, and consecuently the Weight and Resample functions.
    """
    def __init__(self, zf_dim, M, *args) -> None:
        
        self.zf_dim = zf_dim  # dimensionality of a feature observation
        
        self.M = M
        self.nf = len(M)
        super().__init__(*args)


    def Weight(self, z, R): 
        """
        Weight each particle by the liklihood of the particle being correct.
        The probability the particle is correct is given by the probability that it is correct given the measurements (z). 

        
        :param z: measurement vector
        :param R: measurement noise covariance
        :return: None
        """
        # To be completed by the student
        n_particles = len(self.particles)
        n_M         = len(self.M)

        # self.particle_weights = self.particle_weights + 0.02 # evenly distributed weights
        if R == 0:
            print("Standard deviation of the measurement equals to 0, pls change it")

        for inx_particles in range(n_particles):
            for inx_M in range(n_M):
                # Compute distance between one particle to one landmark
                dist_particle_M = np.linalg.norm(self.particles[inx_particles,:2] - self.M[inx_M].T)
                # Compute error between distance above with the measurement
                err = dist_particle_M - z[inx_M]
                # Compute the probability P(z|x) with Gaussian distribution
                pz =  1 / np.sqrt(2*np.pi*R) * np.exp(-(err)**2 / (2 * R))
                # Update the weight of particle
                self.particle_weights[inx_particles] *= pz
            # Normalise weights of all particles
            self.particle_weights /= np.sum(self.particle_weights) 

    def Resample(self):
        """
        Resample the particles based on their weights to ensure diversity and prevent particle degeneracy.

        This function implements the resampling step of a particle filter algorithm. It uses the weights
        assigned to each particle to determine their likelihood of being selected. Particles with higher weights
        are more likely to be selected, while those with lower weights have a lower chance.

        The resampling process helps to maintain a diverse set of particles that better represents the underlying
        probability distribution of the system state. 

        After resampling, the attributes 'particles' and 'weights' of the ParticleFilter instance are updated
        to reflect the new set of particles and their corresponding weights.

        :return: None
        """
        # To be completed by the student
        # Using Low_variance_sampler

        # Sum all particle weights
        W = np.sum(self.particle_weights)
        # Get number of particles weights
        M = len(self.particle_weights)
        # Compute r and initialize c, i value of the Low veriance sampler
        r = W/M * rnd.random()
        c = self.particle_weights[0]
        i = 0
        # Initilize resampled particles
        particles_resampled = np.zeros((len(self.particles),3))

        # Loop
        for i_M in range(M):
            u = r + i_M * W/M
            while u > c:
                i += 1
                c += self.particle_weights[i]   

            particles_resampled[i_M] = self.particles[i]

        # Assign resampled particles to particles
        self.particles = particles_resampled
        # Assign all weights of particles equal to 1/M
        self.particle_weights = np.ones(len(self.particles)) / len(self.particles)

    def Update(self, z, R):
        """
        Update the particle weights based on sensor measurements and perform resampling.

        This function adjusts the weights of particles based on how well they match the sensor measurements.
       
        The updated weights reflect the likelihood of each particle being the true state of the system given
        the sensor measurements.

        After updating the weights, the function may perform resampling to ensure that particles with higher
        weights are more likely to be selected, maintaining diversity and preventing particle degeneracy.
        
        :param z: measurement vector
        :param R: the covariance matrix associated with the measurement vector

        """
        # To be completed by the student      
        N_eff = 1 / np.sum(np.square(self.particle_weights))
        if N_eff < len(self.particle_weights)/2:
            self.Resample()
        else:
            self.Weight(z, R)
            
    def Localize(self):
        uk, Qk =  self.GetInput()

        if uk.size > 0:
            self.Prediction(uk, Qk)
        
        zf, Rf = self.GetMeasurements()
        if zf.size > 0:
            self.Update(zf, Rf)

        self.PlotParticles()
        return self.get_mean_particle()
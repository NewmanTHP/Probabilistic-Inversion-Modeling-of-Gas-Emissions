import sourceinversion.atmospheric_measurements as gp

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
import jax.numpy as jnp
import jax
from jax import random
from jax.flatten_util import ravel_pytree
from dataclasses import dataclass
from jaxtyping import Float
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


__all__ = ['Priors', 'MALA_Within_Gibbs', 'Manifold_MALA_Within_Gibbs', 'Plots']


@dataclass
class Priors:
    """
    MCMC estimated parameter priors.
    """

    # Slab allocation rate prior (used in grid-based inversion)
    theta: Float

    # Log emission rate (log(s)): Log scale Slab and spike prior (used in grid-based inversion)
    log_spike_mean: Float
    log_spike_var: Float
    log_slab_mean: Float
    log_slab_var: Float

    # Source location (source_x, source_y): Set roughly to center of the monitored domain with variance covering the whole domain
    source_location_x_mean: Float
    source_location_y_mean: Float
    source_location_x_var: Float
    source_location_y_var: Float

    # Measurement error variance (sigma squared)
    sigma_squared_con: Float
    sigma_squared_rate: Float

    # Background gas concentration (beta)
    mean_background_prior: Float
    variance_background_prior: Float

    # Dispersion parameter (a_H, a_V, b_H, b_V)
    a_mean: Float
    a_var: Float
    b_mean: Float
    b_var: Float





class Gibbs_samplers:
    """
    This class contains the Gibbs samplers for the conditional posterior distribution of
    background gas concentration, sensor measurement error variance and slab and spike 
    allocation indicator.
    """

    def __init__(self, gaussianplume, data, priors, chilbolton):
        
        self.gaussianplume = gaussianplume
        self.data = data
        self.priors = priors
        self.chilbolton = chilbolton
        if self.chilbolton == True:
            self.number_of_time_steps = int(self.data.shape[0] / 7)
            self.nbr_different_background = 7
        else:
            self.number_of_time_steps = gaussianplume.wind_field.number_of_time_steps
            self.nbr_different_background = gaussianplume.sensors_settings.number_of_different_backgrounds
        self.layout = gaussianplume.sensors_settings.layout


    def background_conditional_posterior(self, s, A,  sigma_squared, key):
        """
        Multivariate Normal conditional posterior distribution sampler for background gas concentration.

        Args:
            s (Array[float]): Emission rate in kg/s.
            A (Array[float]): Gaussian plume model-based coupling matrix in ppm.
            sigma_squared (float): Sensors measurement error variance in ppm.
            key (int): Random key.

        Returns:
            Array[float]: Multivariate Normal sample.
        """

        covariance = 1/((1/ sigma_squared) + (1/self.priors.variance_background_prior))

        if self.layout == "grid":
            a = jnp.array((self.data[::self.number_of_time_steps*self.nbr_different_background] - jnp.matmul(A[::self.number_of_time_steps*self.nbr_different_background], s.reshape(-1,1))) / sigma_squared)
        else:
            a = jnp.array((self.data[::self.number_of_time_steps] - jnp.matmul(A[::self.number_of_time_steps], s.reshape(-1,1))) / sigma_squared)

        b =  self.priors.mean_background_prior / self.priors.variance_background_prior 
        mean = covariance *(a + b)

        return tfd.MultivariateNormalDiag(loc = mean.flatten(), scale_diag = jnp.repeat(jnp.sqrt(covariance), self.nbr_different_background)).sample(sample_shape=(1), seed=key).reshape(-1,1)
    


    def measurement_error_var_conditional_posterior(self, A, beta, s, key):
        """
        Inverse-Gamma conditional posterior for sensor measurement error variance.
        
        Args:
            A (Array[float]): Gaussian plume model-based coupling matrix in ppm.
            beta (Array[float]): Background gas concentration in ppm.
            s (Array[float]): Emission rate in kg/s.
            key (int): Random key.

        Returns:
            (float): Inverse-Gamma sample.
        """

        n = self.gaussianplume.sensors_settings.sensor_number * self.number_of_time_steps
        variance = self.priors.sigma_squared_rate + (jnp.sum(jnp.square(self.data - beta - jnp.matmul(A, s.reshape(-1,1)))) / 2)
        shape = (n/2) + self.priors.sigma_squared_con

        return tfd.InverseGamma(concentration = shape, scale = variance).sample(1, key)[0]



    def binary_indicator_Zi_conditional_posterior(self, log_s, key):
        """
        Binomial conditional posterior distribution sampler for spike and slab binary allocation indicator Zi.

        Args:
            log_s (Array[float]): Log emission rate in kg/s.
            key (int): Random key.

        Returns:
            Array[float]: Binomial sample.
        """

        a = (jnp.square(log_s.reshape(-1,1) - self.priors.log_slab_mean) / (2 * self.priors.log_slab_var)) 
        numerator = jnp.power(2 * jnp.pi * self.priors.log_slab_var, -0.5) *  self.priors.theta
        denominator = numerator + (jnp.power(2 * jnp.pi * self.priors.log_spike_var , -0.5) * jnp.exp(-(jnp.square(log_s.reshape(-1,1) - self.priors.log_spike_mean) / (2 * self.priors.log_spike_var)) + a) * (1-self.priors.theta))
        bern_prob = numerator / denominator

        return tfd.Binomial(total_count=1, probs=bern_prob).sample(1, seed=key).squeeze()





class MWG_tools:
    """
    Contains functions needed for for the Metropolis-within-Gibbs sampler and compatible with Manifold-MALA and MALA.
    """
    def __init__(self, grided, gaussianplume, data, log_posterior, priors, mh_params, fixed, chilbolton, wind_sigmas, step_size_tuning):

        self.grided = grided
        self.gaussianplume = gaussianplume
        self.data = data
        self.log_posterior = log_posterior
        self.priors = priors
        self.mh_unflat_func = ravel_pytree(mh_params)[1]
        self.fixed = fixed
        self.wind_sigmas = wind_sigmas
        self.step_size_tuning = step_size_tuning
        self.chilbolton = chilbolton
        if chilbolton == True:   
            self.fixed_ref1 = fixed[0], fixed[7], fixed[14], fixed[15], fixed[35], fixed[36], fixed[16], fixed[37], fixed[44]
            self.fixed_ref2 = fixed[1], fixed[8], fixed[17], fixed[18], fixed[35], fixed[36], fixed[19], fixed[38], fixed[45]
            self.fixed_ref3 = fixed[2], fixed[9], fixed[20], fixed[21], fixed[35], fixed[36], fixed[22], fixed[39], fixed[46]
            self.fixed_ref4 = fixed[3], fixed[10], fixed[23], fixed[24], fixed[35], fixed[36], fixed[25], fixed[40], fixed[47]
            self.fixed_ref5 = fixed[4], fixed[11], fixed[26], fixed[27], fixed[35], fixed[36], fixed[28], fixed[41], fixed[48]
            self.fixed_ref6 = fixed[5], fixed[12], fixed[29], fixed[30], fixed[35], fixed[36], fixed[31], fixed[42], fixed[49]
            self.fixed_ref7 = fixed[6], fixed[13], fixed[32], fixed[33], fixed[35], fixed[36], fixed[34], fixed[43], fixed[50]
            self.number_of_time_steps = int(self.data.shape[0] / 7)
            self.nbr_different_background = 7
        else:
            self.number_of_time_steps = gaussianplume.wind_field.number_of_time_steps
            self.nbr_different_background = gaussianplume.sensors_settings.number_of_different_backgrounds
        self.layout = gaussianplume.sensors_settings.layout
        self.binary_indicator_Zi_conditional_posterior = Gibbs_samplers(gaussianplume, data, priors, chilbolton).binary_indicator_Zi_conditional_posterior



    
    def first_and_second_output_of_log_posterior(self, *args, **kwargs):
        """
        Wrapper function enabling the function jax.values_and_grad() to compute the log posterior and gradients of the function log_posterior()
        while also returning the second output of the log_posterior() function; the coupling matrix A. This improves the efficiency of the MCMC
        algorithm by avoiding the need to compute the coupling matrix A twice.

        Returns:
            outputs[0] (Array[float]): Log posterior and gradients.
            outputs[1] (Array[float]): Coupling matrix A in ppm.
        """

        outputs = self.log_posterior(*args, **kwargs)

        return outputs[0], outputs[1]

    def glpi(self, x, sigma_squared, betas, s_var, s_mean):
        """
        Evaluates the log posterior and gradients of the log posterior function, and the coupling matrix A.

        Args:
            x (Array[float]): Vector of parameters for gradient-based estimation [log_a_H, log_a_V, log_b_H, log_b_V, log_s, source_x, source_y].
            sigma_squared (float): Sensor measurement error variance in ppm.
            betas (Array[float]): Background gas concentrations in ppm.
            s_var (Array[float]): Log emission rate variance.
            s_mean (Array[float]): Log emission rate mean.
        
        Returns:
            value (Array[float]): Log posterior.
            gradients (Array[float]): Parameter gradients of the log posterior function.
            A (Array[float]): Coupling matrix A in ppm.
        """

        value_and_grad_func = jax.value_and_grad(self.first_and_second_output_of_log_posterior, has_aux=True)
        if self.chilbolton == True:
            (value, second_output), grad = value_and_grad_func(self.mh_unflat_func(x), sigma_squared, betas, s_var, s_mean, self.data, self.priors, self.wind_sigmas, self.number_of_time_steps)
        else:
            (value, second_output), grad = value_and_grad_func(self.mh_unflat_func(x), sigma_squared, betas, s_var, s_mean, self.data, self.priors, self.wind_sigmas)
        return value, grad, second_output


    def mwg_scan(self, step, Gibbs_init, MH_init, iters, r_eps, release_17):
        """
        Metropolis-within-Gibbs sampler using jax.lax.scan() to iterate over the MCMC chain.

        Gibbs sampling of background gas concentration, sensor measurement error variance (and slab and spike
        allocation indicator if grid-based inversion).
        
        Gradient-based sampling of source location and positively constrained dispersion parameters and emission rate.

        Args:
            step (function): MCMC function, either MALA or Manifold-MALA.
            Gibbs_init (dict): Initial values for Gibbs sampling.
            MH_init (dict): Initial values for gradient-based sampling.
            iters (int): Number of MCMC iterations.
            r_eps (float): Initial step size.
            chilbolton (bool): True when using Chilbolton sensor set-up.
            release_17 (bool): True when using the 17th release.
            number_of_time_steps (int): Number of observation times.
            nbr_different_background (int): Number of different background gas concentrations 
                                            i.e number of sensors with different (x,y) locations.
        
        Returns:
            states[0] (Array[float]): Gradient-based parameter samples/chain [log_a_H, log_a_V, log_b_H, log_b_V, log_s, source_x, source_y].
            states[1] (Array[float]): Gibbs samples/chain of sensor measurement error variance.
            states[2] (Array[float]): Gibbs samples/chain of background gas concentration.
            states[3] (Array[float]): Log posterior value chain.
            states[4] (Array[float]): Step size chain. 
            states[5] (Array[float]): Metropolis-Hastings acceptance count chain.
            states[6] (Array[float]): Spike and slab allocation count chain.
            states[9] (Array[float]): Maximum difference between accepted samples and initialisation values of gradient-based parameters.
            states[10] (Array[float]): Gradient chains.
            states[7] (Array[float]): Iteration number chain.
            states[12] (Array[float]): Metropolis-Hastings acceptance rate chain.
        """

        # Fixed seed for reproducibility
        key = random.PRNGKey(7)

        # Sensor measurement error variance
        sigma_squared = Gibbs_init["sigma_squared"]
        
        # Background gas concentration
        background = Gibbs_init["background"].reshape(-1,1)
        if self.chilbolton == True:
            nbr_of_beams = 7
            if release_17 == True:
                # For release 17th in Chilbolton reflector 2 path only has 209 observations
                reflector_2_obs_nbr = 209
                betas = jnp.repeat(background.mean(), (self.number_of_time_steps * (nbr_of_beams-1) ) + reflector_2_obs_nbr).reshape(-1,1)
            else:
                betas = jnp.repeat(background, self.number_of_time_steps).reshape(-1,1)
        elif self.layout == "grid":
            betas = jnp.repeat(background, self.number_of_time_steps*self.nbr_different_background).reshape(-1,1)
        else:
            betas = jnp.repeat(background, self.number_of_time_steps).reshape(-1,1)
        # Emission rate
        if self.grided == True:
            z = self.binary_indicator_Zi_conditional_posterior(self.mh_unflat_func(MH_init)["log_s"], key)
        elif self.grided == False:
            z = jnp.repeat(1, len(self.mh_unflat_func(MH_init)["log_s"]))
        ss_var = jnp.where(z==0, self.priors.log_spike_var, self.priors.log_slab_var).reshape(-1,1)
        ss_mean = jnp.where(z==0, self.priors.log_spike_mean, self.priors.log_slab_mean).reshape(-1,1)
        z_count = jnp.zeros(len(self.mh_unflat_func(MH_init)["log_s"]))
        
        # Log posterior value and gradients
        ll, gradi, _ = self.glpi(MH_init, sigma_squared, betas, ss_var, ss_mean)
        gradients = ravel_pytree(gradi)[0]
        gradients = jnp.where(jnp.isnan(ravel_pytree(gradients)[0]), 0, ravel_pytree(gradients)[0]) # when starting at correct solution gradient can be nan
        
        # Acceptance count
        sum_accept = 0
        
        # Step size
        if self.step_size_tuning == "DOG":
            new_grad_squared_sum = gradients**2
            max_dist = jnp.full(len(MH_init), r_eps)
            dt = max_dist / jnp.sqrt(new_grad_squared_sum+1e-8)
        elif self.step_size_tuning == "Optimal":
            new_grad_squared_sum, max_dist = None, None
            dt = jnp.array([r_eps])
        elif self.step_size_tuning == "False":
            new_grad_squared_sum, max_dist = None, None
            dt = jnp.array([r_eps])
        else:
            raise ValueError("Step size tuning must be either 'DOG', 'Optimal' or 'False'.")

        # Iteration number
        iteration = 0
        _, states = jax.lax.scan(step, [MH_init, sigma_squared, background, ll, dt, sum_accept, z_count, iteration, new_grad_squared_sum, max_dist, gradients, iters, jnp.array(0.0)], jax.random.split(key, iters)) 

        return states[0], states[1], states[2].squeeze(), states[3], states[4], states[5], states[6], states[9], states[10], states[7], states[12]
    




class MALA_Within_Gibbs(gp.GaussianPlume, Priors):

    def __init__(self, grided, gaussianplume, data, log_posterior, priors, mh_params, gibbs_params, fixed, chilbolton, wind_sigmas, release_17, step_size_tuning):
        
        self.grided = grided
        self.gaussianplume = gaussianplume
        self.data = data
        self.log_posterior = log_posterior
        self.priors = priors
        self.mh_params = mh_params
        self.gibbs_params = gibbs_params
        self.mh_flat = ravel_pytree(mh_params)[0]
        self.mh_unflat_func = ravel_pytree(mh_params)[1]
        self.chilbolton = chilbolton
        self.wind_sigmas = wind_sigmas
        self.release_17 = release_17
        self.step_size_tuning = step_size_tuning
        self.fixed = fixed
        if self.chilbolton == True:   
            self.fixed_ref1 = fixed[0], fixed[7], fixed[14], fixed[15], fixed[35], fixed[36], fixed[16], fixed[37], fixed[44]
            self.fixed_ref2 = fixed[1], fixed[8], fixed[17], fixed[18], fixed[35], fixed[36], fixed[19], fixed[38], fixed[45]
            self.fixed_ref3 = fixed[2], fixed[9], fixed[20], fixed[21], fixed[35], fixed[36], fixed[22], fixed[39], fixed[46]
            self.fixed_ref4 = fixed[3], fixed[10], fixed[23], fixed[24], fixed[35], fixed[36], fixed[25], fixed[40], fixed[47]
            self.fixed_ref5 = fixed[4], fixed[11], fixed[26], fixed[27], fixed[35], fixed[36], fixed[28], fixed[41], fixed[48]
            self.fixed_ref6 = fixed[5], fixed[12], fixed[29], fixed[30], fixed[35], fixed[36], fixed[31], fixed[42], fixed[49]
            self.fixed_ref7 = fixed[6], fixed[13], fixed[32], fixed[33], fixed[35], fixed[36], fixed[34], fixed[43], fixed[50]
            self.number_of_time_steps = int(self.data.shape[0] / 7)
            self.nbr_different_background = 7
        else:
            self.number_of_time_steps = gaussianplume.wind_field.number_of_time_steps
            self.nbr_different_background = gaussianplume.sensors_settings.number_of_different_backgrounds
        self.layout = gaussianplume.sensors_settings.layout

        self.background_conditional_posterior = Gibbs_samplers(gaussianplume, data, priors, chilbolton).background_conditional_posterior
        self.measurement_error_var_conditional_posterior = Gibbs_samplers(gaussianplume, data, priors, chilbolton).measurement_error_var_conditional_posterior
        self.binary_indicator_Zi_conditional_posterior = Gibbs_samplers(gaussianplume, data, priors, chilbolton).binary_indicator_Zi_conditional_posterior

        self.glpi = MWG_tools(grided, gaussianplume, data, log_posterior, priors, mh_params, fixed, chilbolton, wind_sigmas, step_size_tuning).glpi
        self.mwg_scan = MWG_tools(grided, gaussianplume, data, log_posterior, priors, mh_params, fixed, chilbolton, wind_sigmas, step_size_tuning).mwg_scan



    def mala_advance(self, x, dt, gradient_x):

        return  x + 0.5*dt*ravel_pytree(gradient_x)[0]



    def mala_rprop(self, key, x, dt, gradient_x):
        """ 
        MALA proposal distribution sampler.

        Args:
            key (int): Random key.
            x (Array[float]): Vector of parameters for gradient-based estimation [log_a_H, log_a_V, log_b_H, log_b_V, log_s, source_x, source_y].
            dt (float): Step size.
            gradient_x (Array[float]): Parameter gradients.

        Returns:
            (Array[float]): MALA proposal distribution sample.
        """

        return self.mala_advance(x, dt, gradient_x) + jnp.sqrt(dt)*jax.random.normal(key, [len(x)])



    def mala_dprop(self, prop, x, dt, gradient_x):
        """
        MALA proposal log probability.

        Args:
            prop (Array[float]): MALA proposal distribution sample.
            x (Array[float]): Vector of parameters for gradient-based estimation [log_a_H, log_a_V, log_b_H, log_b_V, log_s, source_x, source_y].
            dt (float): Step size.
            gradient_x (Array[float]): Parameter gradients.

        Returns:
            (Array[float]): MALA proposal log probability.
        """

        return jnp.sum(tfd.Normal(loc=self.mala_advance(x, dt, gradient_x), scale=jnp.sqrt(dt)).log_prob(prop))



    def mala_step(self, updates, key):
        """
        Mala function to iterate over.

        Args:
            updates (list[Array[float]]): MCMC parameter values and estimated parameter values at the previous iteration.
            key (int): Random key for the current iteration.

        Returns:
            updates (list[Array[float]]): Updated MCMC parameter values and estimated parameter values at the current iteration.
        """

        # Previous iteration values
        [x, sigma_squared, background, current_likelihood, dt, sum_accept, z_count, iteration, new_grad_squared_sum, max_dist, gradients, iters, acceptance_rate] = updates 

        #          Updates          #
        if self.chilbolton == True:
            nbr_of_beams = 7
            if self.release_17 == True:
                # For release 17th in Chilbolton reflector 2 path only has 209 observations
                reflector_2_obs_nbr = 209
                betas = jnp.repeat(background.mean(), (self.number_of_time_steps * (nbr_of_beams-1) ) + reflector_2_obs_nbr).reshape(-1,1)
            else:
                betas = jnp.repeat(background, self.number_of_time_steps).reshape(-1,1)
        elif self.layout == "grid":
            betas = jnp.repeat(background, self.number_of_time_steps*self.nbr_different_background).reshape(-1,1)
        else:
            betas = jnp.repeat(background, self.number_of_time_steps).reshape(-1,1)

        # Emission rate spike and slab if grid-based inversion
        if self.grided == True:
            z = self.binary_indicator_Zi_conditional_posterior(self.mh_unflat_func(x)["log_s"], key)
        elif self.grided == False:
            z = jnp.repeat(1, len(self.mh_unflat_func(x)["log_s"]))
        ss_var = jnp.where(z==0, self.priors.log_spike_var, self.priors.log_slab_var).reshape(-1,1)
        ss_mean = jnp.where(z==0, self.priors.log_spike_mean, self.priors.log_slab_mean).reshape(-1,1)
        
        #           MALA            #
        # Parameter proposal
        current_likelihood, current_gradient, current_A = self.glpi(x, sigma_squared, betas, ss_var, ss_mean)
        prop = self.mala_rprop(key, x, dt, current_gradient)
        proposed_likelihood, proposed_gradient, proposed_A = self.glpi(prop, sigma_squared, betas, ss_var, ss_mean)
        
        # Metropolis-Hastings acceptance step
        a = proposed_likelihood - current_likelihood + self.mala_dprop(x, prop, dt, proposed_gradient) - self.mala_dprop(prop, x, dt, current_gradient)
        a = jnp.where(jnp.isnan(a), -jnp.inf, a)
        accept = (jnp.log(jax.random.uniform(key)) < a)
        new_x, new_ll, new_A = jnp.where(accept, prop, x), jnp.where(accept, proposed_likelihood, current_likelihood), jnp.where(accept, proposed_A, current_A)
        
        #           Gibbs           #
        sigma_squared = self.measurement_error_var_conditional_posterior(new_A, betas, jnp.exp(self.mh_unflat_func(new_x)["log_s"]), key)
        background = self.background_conditional_posterior(jnp.exp(self.mh_unflat_func(new_x)["log_s"]), new_A, sigma_squared, key).reshape(-1,1)

        #          Updates          #
        iteration += 1
        sum_accept += jnp.where(accept, 1, 0)
        acceptance_rate = (0.01*acceptance_rate)+(0.99*sum_accept/iteration) # avoids long memory bias
        z_count += z


        # Step size
        if self.step_size_tuning == "DOG":
            gradients = jnp.where(accept, ravel_pytree(proposed_gradient)[0], ravel_pytree(current_gradient)[0])
            gradients = jnp.where(jnp.isnan(ravel_pytree(gradients)[0]), 0, ravel_pytree(gradients)[0])
            new_grad_squared_sum += gradients**2
            max_dist = jnp.maximum(max_dist,  jnp.sqrt(jnp.square(new_x - self.mh_flat)))
            dt = max_dist / jnp.sqrt(new_grad_squared_sum+1e-8)
        elif self.step_size_tuning == "Optimal":
            dt = dt * ( 1 + 0.1 * ( sum_accept/iteration - 0.574))
        elif self.step_size_tuning == "False":
            pass

        updates = [new_x, sigma_squared, background, new_ll, dt, sum_accept, z_count, iteration, new_grad_squared_sum, max_dist, gradients, iters, acceptance_rate]

        return updates, updates



    def mala_chains(self, Gibbs_init, MH_init, iters, r_eps, release_17 = False):
        """
        MALA within Gibbs sampler.
        
        Args:
            Gibbs_init (dict): Initial values for Gibbs sampling.
            MH_init (dict): Initial values for gradient-based sampling.
            iters (int): Number of MCMC iterations.
            r_eps (float): Initial step size.
            release_17 (bool): True when using the 17th release.
        
        Returns:
            MALA_within_Gibbs_chains (dict): MALA within Gibbs sampler chains.
        """

        # Running and timing the MALA within Gibbs sampler
        t1 = time.time()
        mala_chains = self.mwg_scan(self.mala_step, Gibbs_init, MH_init, iters, r_eps, release_17)
        t2 = time.time()
        running_time = t2-t1
        print("Running time MALA within Gibbs: " + str(round(running_time // 60)) + " minutes " + str(round(running_time % 60)) + " seconds")

        # Extracting the MALA within Gibbs sampler chains
        if self.grided == True:
            if  self.wind_sigmas == True:
                MALA_within_Gibbs_chains = {
                    "a_H": jnp.exp(mala_chains[0][:,0]),
                    "a_V": jnp.exp(mala_chains[0][:,1]),
                    "b_H": jnp.exp(mala_chains[0][:,2]),
                    "b_V": jnp.exp(mala_chains[0][:,3]),
                    "background" : mala_chains[2],
                    "s": jnp.exp(mala_chains[0][:,4:4+len(self.mh_unflat_func(MH_init)["log_s"])]),
                    "sigma_squared": mala_chains[1],
                    "ll": mala_chains[3],
                    "dt": mala_chains[4],
                    "acceptance_rate": mala_chains[5]/jnp.arange(1,iters+1),
                    "z_count": mala_chains[6],
                    "max_dist": mala_chains[7],
                    "gradients": mala_chains[8]
                }
            elif self.wind_sigmas == False:
                MALA_within_Gibbs_chains = {
                    "background" : mala_chains[2],
                    "s": jnp.exp(mala_chains[0][:,:len(self.mh_unflat_func(MH_init)["log_s"])]),
                    "sigma_squared": mala_chains[1],
                    "ll": mala_chains[3],
                    "dt": mala_chains[4],
                    "acceptance_rate": mala_chains[5]/jnp.arange(1,iters+1),
                    "z_count": mala_chains[6],
                    "max_dist": mala_chains[7],
                    "gradients": mala_chains[8]
                }
        elif self.grided == False:
            if  self.wind_sigmas == True:
                MALA_within_Gibbs_chains = {
                    "a_H": jnp.exp(mala_chains[0][:,0]),
                    "a_V": jnp.exp(mala_chains[0][:,1]),
                    "b_H": jnp.exp(mala_chains[0][:,1]),
                    "b_V": jnp.exp(mala_chains[0][:,3]),
                    "background" : mala_chains[2],
                    "s": jnp.exp(mala_chains[0][:, 4 : 4 + len(self.mh_unflat_func(MH_init)["log_s"])]),
                    "sigma_squared": mala_chains[1],
                    "source_x": mala_chains[0][:, 4 + len(self.mh_unflat_func(MH_init)["log_s"]) : 4 + len(self.mh_unflat_func(MH_init)["log_s"]) + len(self.mh_unflat_func(MH_init)["source_x"])],
                    "source_y": mala_chains[0][:, 4 + len(self.mh_unflat_func(MH_init)["log_s"]) + len(self.mh_unflat_func(MH_init)["source_x"]):],
                    "ll": mala_chains[3],
                    "dt": mala_chains[4],
                    "acceptance_rate": mala_chains[5]/jnp.arange(1,iters+1),
                    "z_count": mala_chains[6],
                    "max_dist": mala_chains[7],
                    "gradients": mala_chains[8]
                }
            elif self.wind_sigmas == False:
                MALA_within_Gibbs_chains = {
                    "background" : mala_chains[2],
                    "s": jnp.exp(mala_chains[0][:,:len(self.mh_unflat_func(MH_init)["log_s"])]),
                    "sigma_squared": mala_chains[1],
                    "source_x": mala_chains[0][:,len(self.mh_unflat_func(MH_init)["log_s"]):len(self.mh_unflat_func(MH_init)["log_s"])+len(self.mh_unflat_func(MH_init)["source_x"])],
                    "source_y": mala_chains[0][:,len(self.mh_unflat_func(MH_init)["log_s"])+len(self.mh_unflat_func(MH_init)["source_x"]):],
                    "ll": mala_chains[3],
                    "dt": mala_chains[4],
                    "acceptance_rate": mala_chains[5]/jnp.arange(1,iters+1),
                    "z_count": mala_chains[6],
                    "max_dist": mala_chains[7],
                    "gradients": mala_chains[8]
                }
        
        return MALA_within_Gibbs_chains






class Manifold_MALA_Within_Gibbs(gp.GaussianPlume, Priors):

    def __init__(self, grided, gaussianplume, data, log_posterior, priors, mh_params, gibbs_params, fixed, chilbolton, wind_sigmas, release_17, step_size_tuning):
        
        self.grided = grided
        self.gaussianplume = gaussianplume
        self.data = data
        self.log_posterior = log_posterior
        self.priors = priors
        self.mh_params = mh_params
        self.gibbs_params = gibbs_params
        self.mh_flat = ravel_pytree(mh_params)[0]
        self.mh_unflat_func = ravel_pytree(mh_params)[1]
        self.chilbolton = chilbolton
        self.wind_sigmas = wind_sigmas
        self.release_17 = release_17
        self.step_size_tuning = step_size_tuning
        self.fixed = fixed
        if self.chilbolton == True:   
            self.fixed_ref1 = fixed[0], fixed[7], fixed[14], fixed[15], fixed[35], fixed[36], fixed[16], fixed[37], fixed[44]
            self.fixed_ref2 = fixed[1], fixed[8], fixed[17], fixed[18], fixed[35], fixed[36], fixed[19], fixed[38], fixed[45]
            self.fixed_ref3 = fixed[2], fixed[9], fixed[20], fixed[21], fixed[35], fixed[36], fixed[22], fixed[39], fixed[46]
            self.fixed_ref4 = fixed[3], fixed[10], fixed[23], fixed[24], fixed[35], fixed[36], fixed[25], fixed[40], fixed[47]
            self.fixed_ref5 = fixed[4], fixed[11], fixed[26], fixed[27], fixed[35], fixed[36], fixed[28], fixed[41], fixed[48]
            self.fixed_ref6 = fixed[5], fixed[12], fixed[29], fixed[30], fixed[35], fixed[36], fixed[31], fixed[42], fixed[49]
            self.fixed_ref7 = fixed[6], fixed[13], fixed[32], fixed[33], fixed[35], fixed[36], fixed[34], fixed[43], fixed[50]
            self.number_of_time_steps = int(self.data.shape[0] / 7)
            self.nbr_different_background = 7
        else:
            self.number_of_time_steps = gaussianplume.wind_field.number_of_time_steps
            self.nbr_different_background = gaussianplume.sensors_settings.number_of_different_backgrounds
        self.layout = gaussianplume.sensors_settings.layout

        self.background_conditional_posterior = Gibbs_samplers(gaussianplume, data, priors, chilbolton).background_conditional_posterior
        self.measurement_error_var_conditional_posterior = Gibbs_samplers(gaussianplume, data, priors, chilbolton).measurement_error_var_conditional_posterior
        self.binary_indicator_Zi_conditional_posterior = Gibbs_samplers(gaussianplume, data, priors, chilbolton).binary_indicator_Zi_conditional_posterior

        self.glpi = MWG_tools(grided, gaussianplume, data, log_posterior, priors, mh_params, fixed, chilbolton, wind_sigmas, step_size_tuning).glpi
        self.mwg_scan = MWG_tools(grided, gaussianplume, data, log_posterior, priors, mh_params, fixed, chilbolton, wind_sigmas, step_size_tuning).mwg_scan


    def extract_values(self, d):
        if isinstance(d, dict):
            for v in d.values():
                yield from self.extract_values(v)
        else:
            yield d


    def inverse_hessian(self, x, sigma_squared, betas, ss_var, ss_mean):
        """
        Computes the inverse Hessian matrix.

        Args:
            x (Array[float]): Vector of parameters for gradient-based estimation [log_a_H, log_a_V, log_b_H, log_b_V, log_s, source_x, source_y].
            sigma_squared (float): Sensor measurement error variance in ppm.
            betas (Array[float]): Background gas concentrations in ppm.
            ss_var (Array[float]): Log emission rate variance.
            ss_mean (Array[float]): Log emission rate mean.

        Returns:
            (Array[float]): Inverse Hessian matrix.
        """
        if self.chilbolton == True:
            hess = jax.jacfwd(jax.jacrev(self.log_posterior, has_aux=True), has_aux=True)(self.mh_unflat_func(x), sigma_squared, betas, ss_var, ss_mean, self.data, self.priors, self.wind_sigmas, self.number_of_time_steps)[0]
        else:
            hess = jax.jacfwd(jax.jacrev(self.log_posterior, has_aux=True), has_aux=True)(self.mh_unflat_func(x), sigma_squared, betas, ss_var, ss_mean, self.data, self.priors, self.wind_sigmas)[0]
        values = list(self.extract_values(hess))
        negative_hessians = []
        [negative_hessians.append(i.flatten()) for i in values]
        negative_hessians = jnp.concatenate(negative_hessians).reshape(len(x),len(x))
        inv_negative_hessian = jnp.linalg.inv(negative_hessians)

        return - inv_negative_hessian



    def sqrt_inv_hess(self, inv_hessian):
        """
        Computes the square root of the inverse Hessian matrix.

        Args:
            inv_hessian (Array[float]): Inverse Hessian matrix.

        Returns:
            (Array[float]): Square root of the inverse Hessian matrix.
        """

        # Compute the eigenvalues and eigenvectors of the inverse Hessian
        eigenvalues, eigenvectors = jnp.linalg.eigh(inv_hessian)
        # Compute the square root of the absolute value of the eigenvalues
        sqrt_eigenvalues = jnp.sqrt(jnp.abs(eigenvalues))
        # Compute the square root of the inverse Hessian
        sqrt_inv_hessian = eigenvectors @ jnp.diag(sqrt_eigenvalues) @ jnp.linalg.inv(eigenvectors)

        return sqrt_inv_hessian



    def manifold_mala_advance(self, x, dt, inv_hessian_x, gradient_x):

        return  x.reshape(-1,1) + 0.5*dt*jnp.matmul(inv_hessian_x, ravel_pytree(gradient_x)[0].reshape(-1,1))



    def manifold_mala_rprop(self, key, x, dt, inv_hessian_x, gradient_x):
        """
        Manifold-MALA proposal distribution sampler.

        Args:
            key (int): Random key.
            x (Array[float]): Vector of parameters for gradient-based estimation [log_a_H, log_a_V, log_b_H, log_b_V, log_s, source_x, source_y].
            dt (float): Step size.
            inv_hessian_x (Array[float]): Inverse Hessian matrix.
            gradient_x (Array[float]): Parameter gradients.

        Returns:
            (Array[float]): Manifold-MALA proposal distribution sample.
        """

        return (self.manifold_mala_advance(x, dt, inv_hessian_x, gradient_x) + (jnp.sqrt(dt)*self.sqrt_inv_hess(inv_hessian_x)@jax.random.normal(key, [len(x)])).reshape(-1,1)).flatten()



    def manifold_mala_dprop(self, prop, x, dt, inv_hessian_x, gradient_x):
        """
        Manifold-MALA proposal log probability.

        Args:
            prop (Array[float]): Manifold-MALA proposal distribution sample.
            x (Array[float]): Vector of parameters for gradient-based estimation [log_a_H, log_a_V, log_b_H, log_b_V, log_s, source_x, source_y].
            dt (float): Step size.
            inv_hessian_x (Array[float]): Inverse Hessian matrix.
            gradient_x (Array[float]): Parameter gradients.

        Returns:
            (Array[float]): Manifold-MALA proposal log probability.
        """

        return tfd.MultivariateNormalFullCovariance(loc=self.manifold_mala_advance(x, dt, inv_hessian_x, gradient_x).flatten(), covariance_matrix=jnp.array(jnp.sqrt(dt)*self.sqrt_inv_hess(inv_hessian_x))).log_prob(prop.flatten())



    def manifold_mala_step(self, updates, key):
        """
        Manifold-MALA function to iterate over.

        Args:
            updates (list[Array[float]]): MCMC parameter values and estimated parameter values at the previous iteration.
            key (int): Random key for the current iteration.

        Returns:
            updates (list[Array[float]]): Updated MCMC parameter values and estimated parameter values at the current iteration
        """

        # Previous iteration values
        [x, sigma_squared, background, current_likelihood, dt, sum_accept, z_count, iteration, new_grad_squared_sum, max_dist, gradients, iters, acceptance_rate] = updates

        #          Updates          #
        if self.chilbolton == True:
            nbr_of_beams = 7
            if self.release_17 == True:
                # For release 17th in Chilbolton reflector 2 path only has 209 observations
                reflector_2_obs_nbr = 209
                betas = jnp.repeat(background.mean(), (self.number_of_time_steps * (nbr_of_beams-1) ) + reflector_2_obs_nbr).reshape(-1,1)
            else:
                betas = jnp.repeat(background, self.number_of_time_steps).reshape(-1,1)
        elif self.layout == "grid":
            betas = jnp.repeat(background, self.number_of_time_steps*self.nbr_different_background).reshape(-1,1)
        else:
            betas = jnp.repeat(background, self.number_of_time_steps).reshape(-1,1)

        # Emission rate spike and slab if grid-based inversion
        if self.grided == True:
            z = self.binary_indicator_Zi_conditional_posterior(self.mh_unflat_func(x)["log_s"], key)
        elif self.grided == False:
            z = jnp.repeat(1, len(self.mh_unflat_func(x)["log_s"]))
        ss_var = jnp.where(z==0, self.priors.log_spike_var, self.priors.log_slab_var).reshape(-1,1)
        ss_mean = jnp.where(z==0, self.priors.log_spike_mean, self.priors.log_slab_mean).reshape(-1,1)
        
        #          M-MALA            #
        current_likelihood, current_gradient, current_A = self.glpi(x, sigma_squared, betas, ss_var, ss_mean)
        current_inv_hessian = self.inverse_hessian(x, sigma_squared, betas, ss_var, ss_mean)
        prop = self.manifold_mala_rprop(key, x, dt, current_inv_hessian, current_gradient)
        proposed_likelihood, proposed_gradient, proposed_A = self.glpi(prop, sigma_squared, betas, ss_var, ss_mean)
        proposed_inv_hessian = self.inverse_hessian(prop, sigma_squared, betas, ss_var, ss_mean)

        a = proposed_likelihood - current_likelihood + self.manifold_mala_dprop(x, prop, dt, proposed_inv_hessian, proposed_gradient) - self.manifold_mala_dprop(prop, x, dt, current_inv_hessian, current_gradient)
        
        a = jnp.where(jnp.isnan(a), -jnp.inf, a)
        accept = (jnp.log(jax.random.uniform(key)) < a)
        new_x, new_ll, new_A = jnp.where(accept, prop, x), jnp.where(accept, proposed_likelihood, current_likelihood), jnp.where(accept, proposed_A, current_A)
        
        #           Gibbs           #
        sigma_squared = self.measurement_error_var_conditional_posterior(new_A, betas, jnp.exp(self.mh_unflat_func(new_x)["log_s"]), key)
        background = self.background_conditional_posterior(jnp.exp(self.mh_unflat_func(new_x)["log_s"]), new_A, sigma_squared, key).reshape(-1,1)

        #          Updates          #
        iteration += 1
        sum_accept += jnp.where(accept, 1, 0)
        acceptance_rate = (0.01*acceptance_rate)+(0.99*sum_accept/iteration)
        z_count += z

        # Step size
        if self.step_size_tuning == "DOG":
            gradients = jnp.where(accept, ravel_pytree(proposed_gradient)[0], ravel_pytree(current_gradient)[0])
            gradients = jnp.where(jnp.isnan(ravel_pytree(gradients)[0]), 0, ravel_pytree(gradients)[0])
            new_grad_squared_sum += gradients**2
            max_dist = jnp.maximum(max_dist,  jnp.sqrt(jnp.square(new_x - self.mh_flat)))
            dt = max_dist / jnp.sqrt(new_grad_squared_sum+1e-8)
        elif self.step_size_tuning == "Optimal":
            dt = dt * ( 1 + 0.1 * ( sum_accept/iteration - 0.574))
        elif self.step_size_tuning == "False":
            pass


        updates = [new_x, sigma_squared, background, new_ll, dt, sum_accept, z_count, iteration, new_grad_squared_sum, max_dist, gradients, iters, acceptance_rate]

        return updates, updates



    def manifold_mala_chains(self, Gibbs_init, MH_init, iters, r_eps, release_17 = False):
        """
        Manifold MALA within Gibbs sampler.

        Args:
            Gibbs_init (dict): Initial values for Gibbs sampling.
            MH_init (dict): Initial values for gradient-based sampling.
            iters (int): Number of MCMC iterations.
            r_eps (float): Initial step size.
            release_17 (bool): True when using the 17th release.

        Returns:
            Manifold_MALA_within_Gibbs_chains (dict): Manifold MALA within Gibbs sampler chains.
        """

        # Running and timing the Manifold MALA within Gibbs sampler
        t1 = time.time()
        manifold_mala_chains = self.mwg_scan(self.manifold_mala_step, Gibbs_init, MH_init, iters, r_eps, release_17)
        t2 = time.time()
        running_time = t2-t1
        print("Running time Manifold MALA within Gibbs: " + str(round(running_time // 60)) + " minutes " + str(round(running_time % 60)) + " seconds")

        # Extracting the Manifold MALA within Gibbs sampler chains
        if self.grided == True:
            if  self.wind_sigmas == True:
                Manifold_MALA_within_Gibbs_chains = {
                "a_H": jnp.exp(manifold_mala_chains[0][:,0]),
                "a_V": jnp.exp(manifold_mala_chains[0][:,1]),
                "b_H": jnp.exp(manifold_mala_chains[0][:,2]),
                "b_V": jnp.exp(manifold_mala_chains[0][:,3]),
                "background" : manifold_mala_chains[2],
                "s": jnp.exp(manifold_mala_chains[0][:,4:4+len(self.mh_unflat_func(MH_init)["log_s"])]),
                "sigma_squared": manifold_mala_chains[1],
                "ll": manifold_mala_chains[3],
                "dt": manifold_mala_chains[4],
                "acceptance_rate": manifold_mala_chains[10],
                "z_count": manifold_mala_chains[6],
                "max_dist": manifold_mala_chains[7],
                'gradients': manifold_mala_chains[8],
                }
            elif self.wind_sigmas == False:
                Manifold_MALA_within_Gibbs_chains = {
                "background" : manifold_mala_chains[2],
                "s": jnp.exp(manifold_mala_chains[0][:,0:len(self.mh_unflat_func(MH_init)["log_s"])]),
                "sigma_squared": manifold_mala_chains[1],
                "ll": manifold_mala_chains[3],
                "dt": manifold_mala_chains[4],
                "acceptance_rate": manifold_mala_chains[10],
                "z_count": manifold_mala_chains[6],
                "max_dist": manifold_mala_chains[7],
                "gradients": manifold_mala_chains[8]
                }
        elif self.grided == False:
            if self.wind_sigmas == True:
                Manifold_MALA_within_Gibbs_chains = {
                "a_H": jnp.exp(manifold_mala_chains[0][:,0]),
                "a_V": jnp.exp(manifold_mala_chains[0][:,1]),
                "b_H": jnp.exp(manifold_mala_chains[0][:,2]),
                "b_V": jnp.exp(manifold_mala_chains[0][:,3]),
                "background" : manifold_mala_chains[2],
                "s": jnp.exp(manifold_mala_chains[0][:,4:4+len(self.mh_unflat_func(MH_init)["log_s"])]),
                "sigma_squared": manifold_mala_chains[1],
                "source_x": manifold_mala_chains[0][:, 4 + len(self.mh_unflat_func(MH_init)["log_s"]) : 4 + len(self.mh_unflat_func(MH_init)["log_s"]) + len(self.mh_unflat_func(MH_init)["source_x"])],
                "source_y": manifold_mala_chains[0][:, 4 + len(self.mh_unflat_func(MH_init)["log_s"]) + len(self.mh_unflat_func(MH_init)["source_x"]):],
                "ll": manifold_mala_chains[3],
                "dt": manifold_mala_chains[4],
                "acceptance_rate": manifold_mala_chains[10],
                "z_count": manifold_mala_chains[6],
                "max_dist": manifold_mala_chains[7],
                'gradients': manifold_mala_chains[8],
                }
            elif self.wind_sigmas == False:
                Manifold_MALA_within_Gibbs_chains = {
                "background" : manifold_mala_chains[2],
                "s": jnp.exp(manifold_mala_chains[0][:,0:len(self.mh_unflat_func(MH_init)["log_s"])]),
                "sigma_squared": manifold_mala_chains[1],
                "source_x": manifold_mala_chains[0][:, len(self.mh_unflat_func(MH_init)["log_s"]):len(self.mh_unflat_func(MH_init)["log_s"])+len(self.mh_unflat_func(MH_init)["source_x"])],
                "source_y": manifold_mala_chains[0][:, len(self.mh_unflat_func(MH_init)["log_s"])+len(self.mh_unflat_func(MH_init)["source_x"]):],
                "ll": manifold_mala_chains[3],
                "dt": manifold_mala_chains[4],
                "acceptance_rate": manifold_mala_chains[10],
                "z_count": manifold_mala_chains[6],
                "max_dist": manifold_mala_chains[7],
                "gradients": manifold_mala_chains[8]
                }
        return Manifold_MALA_within_Gibbs_chains





class Plots:

    def __init__(self, gaussianplume, truth):
        self.gaussianplume = gaussianplume
        self.truth = truth



    def sources_emission_rates_chains(self, chains, truth = None, grided = False, save = False, format = "pdf", simulation=True):
        plt.figure(figsize=(15,5))
        if grided == True:
            plt.plot(chains["s"][:,jnp.where(self.truth[2]>0)[0]])
        elif grided == False:
            plt.plot(chains["s"][:,:])

        if simulation == True:
            plt.axhline(self.gaussianplume.atmospheric_state.emission_rate, color='red', linestyle='--')
        elif simulation == False:
            plt.axhline(truth, color='red', linestyle='--')

        plt.title("Source emission rate")
        plt.xlabel("Iterations")
        plt.ylabel("Emission rate (kg/s)")
        if save:
            plt.savefig("true_source_emission_rate." + format, dpi=300, bbox_inches="tight")
        return  plt.show()
    
    def sources_emission_rates_density(self, chains, burn_in = 0, truth = None, grided = False, save = False, format = "pdf", simulation=True):
        plt.figure(figsize=(7,5))
        if grided == True:
            sns.kdeplot(chains["s"][burn_in:,jnp.where(self.truth[2]>0)[0]].squeeze(), color='seagreen', label='MALA', fill=True, alpha=0.1)
        elif grided == False:
            sns.kdeplot(chains["s"][burn_in:,:].squeeze(), color='seagreen', label='MALA', fill=True, alpha=0.1)

        if simulation == True:
            plt.axvline(self.gaussianplume.atmospheric_state.emission_rate, color='red', linestyle='--')
        elif simulation == False:
            plt.axvline(truth, color='red', linestyle='--')
        plt.xlabel("Emission rate (kg/s)")
        plt.ylabel("Density")
        plt.title("Source emission rate")
        plt.legend()
        if save:
            plt.savefig("true source emission rate density." + format, dpi=300, bbox_inches="tight")
        return  plt.show()



    def total_emission_rates_chains(self, chains, save = False, format = "pdf"):
        plt.plot(jnp.sum(chains["s"][:,:], axis=1))
        plt.title("Total Emissions Rates Over Grid")
        plt.axhline(self.gaussianplume.atmospheric_state.emission_rate, color='red', linestyle='--')
        if save:
            plt.savefig("total emission rates." + format, dpi=300, bbox_inches="tight")
        plt.show()

    def total_emission_rates_density(self, chains, burn_in, save = False, format = "pdf"):
        sns.kdeplot(jnp.sum(chains["s"][burn_in:,:], axis=1), color='seagreen', fill=True, alpha=0.1)
        plt.axvline(self.gaussianplume.atmospheric_state.emission_rate, color='red', linestyle='--')
        plt.xlabel("Total Emissions Rate")
        plt.ylabel("Density")
        plt.title("Total Emissions Rates Over Grid")
        plt.legend()
        if save:
            plt.savefig("total emission rates density." + format, dpi=300, bbox_inches="tight")
        return  plt.show()



    def zero_emission_rates_chains(self, chains, save = False, format = "pdf"):
        plt.plot(jnp.delete(chains["s"][:], jnp.where(self.truth[2]>0)[0], axis=1))
        plt.title("Zero Emissions source")
        if save:
            plt.savefig("Zero emissions source." + format, dpi=300, bbox_inches="tight")
        plt.show()



    def measurement_error_var_chains(self, chains, truth = None, save = False, format = "pdf", simulation = True):
        plt.figure(figsize=(15,5))
        plt.plot(chains["sigma_squared"][:])
        plt.title("Measurement error variance")
        plt.xlabel("Iterations")
        plt.ylabel("Sigma squared")

        if simulation == True:
            plt.axhline(self.gaussianplume.sensors_settings.measurement_error_var, color='red', linestyle='--')
        
        plt.legend(["MALA", "True value"])

        if save:
            plt.savefig("sigma squared." + format, dpi=300, bbox_inches="tight")
        plt.show()

    def measurement_error_var_density(self, chains, burn_in = 0, truth = None, save = False, format = "pdf", simulation = True):
        plt.figure(figsize=(7,5))
        plt.title("Sigma squared")
        plt.xlabel("Sigma squared")
        plt.ylabel("Density")
        sns.kdeplot(chains["sigma_squared"][burn_in:], color='seagreen', fill=True, alpha=0.1)

        if simulation == True:
            plt.axvline(self.gaussianplume.sensors_settings.measurement_error_var, color='red', linestyle='--')

        plt.legend(["MALA", "True value"])
        
        if save:
            plt.savefig("sigma squared density." + format, dpi=300, bbox_inches="tight")
        return  plt.show()



    def background_chains(self, chains, burn_in = 0, save = False, format = "pdf", chilbolton = False):
        plt.plot(chains["background"][burn_in:, :])
        plt.title("Background")
        for i in jnp.unique(self.truth[4]):
            plt.axhline(i, linestyle='--')
        if save:
            plt.savefig("background." + format, dpi=300, bbox_inches="tight")
        plt.show()
    
    def background_density(self, chains, burn_in, save = False, format = "pdf", chilbolton = False):
        sns.kdeplot(chains["background"][burn_in:], color='seagreen', fill=True, alpha=0.1)
        plt.axvline(jnp.unique(self.truth[4]), linestyle='--')
        plt.xlabel("Background")
        plt.ylabel("Density")
        plt.title("Background")
        plt.legend()
        if save:
            plt.savefig("background density." + format, dpi=300, bbox_inches="tight")
        return  plt.show()



    def tan_gamma_H_chains(self, chains, save = False, format = "pdf", chilbolton = False):
        plt.plot(chains["a_H"][:])
        plt.title("a_H")
        if chilbolton == False:
            plt.axhline(jnp.tan(self.gaussianplume.atmospheric_state.horizontal_angle), color='red', linestyle='--')
        if save:
            plt.savefig("a_H." + format, dpi=300, bbox_inches="tight")
        plt.show()

    def tan_gamma_H_density(self, chains, burn_in, save = False, format = "pdf", chilbolton = False):
        sns.kdeplot(chains["a_H"][burn_in:], color='seagreen', fill=True, alpha=0.1)
        if chilbolton == False:
            plt.axvline(jnp.tan(self.gaussianplume.atmospheric_state.horizontal_angle), color='red', linestyle='--')
        plt.xlabel("a_H")
        plt.ylabel("Density")
        plt.title("a_H")
        plt.legend()
        if save:
            plt.savefig("tan_gamma_H density." + format, dpi=300, bbox_inches="tight")
        return  plt.show()
    


    def tan_gamma_V_chains(self, chains, save = False, format = "pdf", chilbolton = False):
        plt.plot(chains["a_V"][:])
        plt.title("a_V")
        if chilbolton == False:
            plt.axhline(jnp.tan(self.gaussianplume.atmospheric_state.vertical_angle), color='red', linestyle='--')
        if save:
            plt.savefig("a_V." + format, dpi=300, bbox_inches="tight")
        plt.show()

    def tan_gamma_V_density(self, chains, burn_in, save = False, format = "pdf", chilbolton = False):
        sns.kdeplot(chains["a_V"][burn_in:], color='seagreen', fill=True, alpha=0.1)
        if chilbolton == False:
            plt.axvline(jnp.tan(self.gaussianplume.atmospheric_state.vertical_angle), color='red', linestyle='--')
        plt.xlabel("a_V")
        plt.ylabel("Density")
        plt.title("a_V")
        plt.legend()
        if save:
            plt.savefig("tan_gamma_V density." + format, dpi=300, bbox_inches="tight")
        return  plt.show()
    


    def b_H_chains(self, chains, save = False, format = "pdf", chilbolton = False):
        plt.plot(chains["b_H"][:])
        plt.title("b_H")
        if chilbolton == False:
            plt.axhline(self.gaussianplume.atmospheric_state.downwind_power_H, color='red', linestyle='--')
        if save:
            plt.savefig("b_H." + format, dpi=300, bbox_inches="tight")
        plt.show()

    def b_H_density(self, chains, burn_in, save = False, format = "pdf", chilbolton = False):
        sns.kdeplot(chains["b_H"][burn_in:], color='seagreen', fill=True, alpha=0.1)
        if chilbolton == False:
            plt.axvline(self.gaussianplume.atmospheric_state.downwind_power_H, color='red', linestyle='--')
        plt.xlabel("b_H")
        plt.ylabel("Density")
        plt.title("b_H")
        plt.legend()
        if save:
            plt.savefig("b_H density." + format, dpi=300, bbox_inches="tight")
        return  plt.show()
    


    def b_V_chains(self, chains, save = False, format = "pdf", chilbolton = False):
        plt.plot(chains["b_V"][:])
        plt.title("b_V")
        if chilbolton == False:
            plt.axhline(self.gaussianplume.atmospheric_state.downwind_power_V, color='red', linestyle='--')
        if save:
            plt.savefig("b_V." + format, dpi=300, bbox_inches="tight")
        plt.show()

    def b_V_density(self, chains, burn_in, save = False, format = "pdf", chilbolton = False):
        sns.kdeplot(chains["b_V"][burn_in:], color='seagreen', fill=True, alpha=0.1)
        if chilbolton == False:
            plt.axvline(self.gaussianplume.atmospheric_state.downwind_power_V, color='red', linestyle='--')
        plt.xlabel("b_V")
        plt.ylabel("Density")
        plt.title("b_V")
        plt.legend()
        if save:
            plt.savefig("b_V density." + format, dpi=300, bbox_inches="tight")
        return  plt.show()
    


    def source_position_chains(self, chains, save = False, format = "pdf", chilbolton = False):
        plt.plot(chains["source_x"], chains["source_y"])
        if chilbolton == False:
            plt.scatter(self.gaussianplume.source_location.source_location_x, self.gaussianplume.source_location.source_location_y, color='red', marker='x')
        else:
            plt.scatter(58, 58, color='black', marker='x')
        plt.title("Source Position")
        if save:
            plt.savefig("source position." + format, dpi=300, bbox_inches="tight")
        plt.show()
        
    def source_position_density(self, chains, burn_in, source_loc = None, save = False, format = "pdf", chilbolton = False):
        sns.kdeplot(x=chains["source_x"][burn_in:,0], y=chains["source_y"][burn_in:,0], cmap='rainbow', fill=True, alpha=0.8)
        if chains["source_x"].shape[1] > 1:
            sns.kdeplot(x=chains["source_x"][burn_in:,1], y=chains["source_y"][burn_in:,1], cmap='viridis', fill=True, alpha=0.8)
        if chilbolton == False:
            plt.scatter(self.gaussianplume.source_location.source_location_x, self.gaussianplume.source_location.source_location_y, color='red', marker='x')
        elif chilbolton == True:
            if source_loc is not None:
                for i in range(source_loc.shape[1]):
                    plt.scatter(source_loc[0,i], source_loc[1,i], color='black', marker='x')
            plt.scatter(60, 100, color='red', marker='.')
            plt.scatter(43.94155595876509, 69.06582317966968, color='pink', marker='*')
            plt.scatter(41.1003610083113004, 38.103283315896100, color='pink', marker='*')
            plt.scatter(52.96603569283616, 55.969483627937734, color='pink', marker='*')
            plt.scatter(45.927837100838439, 2.9772519853003425, color='pink', marker='*')
            plt.scatter(60.987300214696843, 16.893100623764694, color='pink', marker='*')
            plt.scatter(66.03007201925851, 42.03314867243171, color='pink', marker='*')
            plt.scatter(66.9186939172796, 67.08154541626573, color='pink', marker='*')
        plt.xlabel("Source x")
        plt.ylabel("Source y")
        plt.title("Source Position Density")
        if save:
            plt.savefig("source position density." + format, dpi=300, bbox_inches="tight")
        
        return plt.show()



    def emission_rates_heatmap(self, chains, burn_in, save = False, format = "pdf"):
        df = pd.DataFrame(jnp.mean(chains["s"][burn_in:,:], axis=0).reshape(len(self.gaussianplume.grid.x), len(self.gaussianplume.grid.y)))
        plt.figure(figsize=(15, 10))
        if (len(self.gaussianplume.grid.x) > 10) and (len(self.gaussianplume.grid.y) > 10):
            ax = sns.heatmap(df, cmap="viridis", xticklabels=round(len(self.gaussianplume.grid.x)/10), yticklabels=round(len(self.gaussianplume.grid.y)/10))
            ax.set_xticks(jnp.arange(0, len(self.gaussianplume.grid.x) + 1, round(len(self.gaussianplume.grid.x)/10)))
            ax.set_yticks(jnp.arange(0, len(self.gaussianplume.grid.y) + 1, round(len(self.gaussianplume.grid.y)/10)))
            ax.set_yticklabels([round(i,1) for i in jnp.arange(self.gaussianplume.grid.y_range[0], self.gaussianplume.grid.y_range[1] + self.gaussianplume.grid.dy + 1, self.gaussianplume.grid.y_range[1]/10)], rotation=0)
            ax.set_xticklabels([round(i,1) for i in jnp.arange(self.gaussianplume.grid.x_range[0], self.gaussianplume.grid.x_range[1] + self.gaussianplume.grid.dx + 1, self.gaussianplume.grid.x_range[1]/10)], rotation=0)
        else:
            ax = sns.heatmap(df, cmap="viridis")
            ax.set_yticklabels(self.gaussianplume.grid.y)
            ax.set_xticklabels(self.gaussianplume.grid.x)
        ax.invert_yaxis()
        ax.scatter(float(self.gaussianplume.source_location.source_location_x/self.gaussianplume.grid.dx), float(self.gaussianplume.source_location.source_location_y/self.gaussianplume.grid.dy), marker='.', s=300, color='orange')
        colorbar = ax.collections[0].colorbar
        colorbar.set_label('Parts per million (PPM)')
        plt.title("Initial Gaussian Plume")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        if save == True:
            plt.savefig("Initial Gaussian Plume." + format, dpi=300, bbox_inches="tight")

        return plt.show()



    def step_size_chains(self, chains, save = False, format = "pdf"):
        plt.plot(chains["dt"][:])
        plt.title("Step Size")
        if save:
            plt.savefig("step size." + format, dpi=300, bbox_inches="tight")
        plt.show()



    def samples_acceptance_rate(self, chains, save = False, format = "pdf"):
        plt.plot(chains["acceptance_rate"][:])
        plt.title("Acceptance Rate")
        if save:
            plt.savefig("acceptance rate." + format, dpi=300, bbox_inches="tight")
        plt.show()



    def spike_slab_allocation(self, chains, save = False, format = "pdf"):
        plt.plot(chains["z_count"][:])
        plt.title("Spike Slab Allocation")
        if save:
            plt.savefig("spike slab allocation." + format, dpi=300, bbox_inches="tight")
        plt.show()



    def log_posterior_plot(self, chains, save = False, format = "pdf"):
        plt.plot(chains["ll"][:])
        plt.title("Log Posterior")
        if save:
            plt.savefig("log posterior." + format, dpi=300, bbox_inches="tight")
        plt.show()


    

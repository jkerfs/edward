#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Normal, Beta, Bernoulli, PointMass

ed.set_seed(42)
np.random.seed(42)


class test_map_class(tf.test.TestCase):

    def test_normal_normal(self):
        N = 10

        # True Latent Parameter Values
        mu_true = 2.0
        sigma_true = 1.0

        # DATA
        x_data = np.random.normal(loc=mu_true, scale=sigma_true, size=N)

        # PRIOR HYPERPARAMETERS
        mu_mu_prior = 1.0
        mu_sigma_prior = 1.0

        with self.test_session() as sess:
            # MODEL: Normal-Normal with known variance
            mu = Normal(mu=mu_mu_prior, sigma=mu_sigma_prior)
            x = Normal(mu=tf.ones(N) * mu, sigma=sigma_true)

            data = {x: x_data}
            inference = ed.MAP([mu], data)
            inference.run()

            # MAP SOLUTION
            mu_posterior_mode_map = sess.run(inference.latent_vars[mu].mean())

            # ANALYTICAL SOLUTION
            mu_sample = np.mean(x_data)
            denom = (N * mu_sigma_prior ** 2 + sigma_true ** 2)
            prior_scale = sigma_true ** 2 / denom
            sample_scale = N * mu_sigma_prior ** 2 / denom
            mu_posterior_mode_analytical = prior_scale * mu_mu_prior + sample_scale * mu_sample

            self.assertAlmostEqual(mu_posterior_mode_analytical, mu_posterior_mode_map, places=4)

    def test_beta_bernoulli(self):
        N = 10

        # DATA
        x_data = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])

        # PRIOR HYPERPARAMETERS
        p_a_prior = 1.0
        p_b_prior = 1.0

        with self.test_session() as sess:
            # MODEL: Normal-Normal with known variance
            p = Beta(a=p_a_prior, b=p_b_prior)
            x = Bernoulli(p=tf.ones(N) * p)

            qp_params = tf.sigmoid(tf.Variable(tf.random_normal([])))
            qp = PointMass(params=qp_params)

            data = {x: x_data}
            inference = ed.MAP({p: qp}, data)
            inference.run()

            # MAP SOLUTION
            p_posterior_mode_map = sess.run(inference.latent_vars[p].mean())

            # ANALYTICAL SOLUTION
            a_posterior_analytical = p_a_prior + np.count_nonzero(x_data)
            b_posterior_analytical = p_b_prior + (np.size(x_data) - np.count_nonzero(x_data))
            p_posterior_mode_analytical = (a_posterior_analytical - 1) /\
                                          (a_posterior_analytical + b_posterior_analytical - 2)

            self.assertAlmostEqual(p_posterior_mode_analytical, p_posterior_mode_map, places=4)

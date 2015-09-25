"""
@author: Ke Zhai (zhaike@umd.edu)

This code was modified from the code originally written by Frank Wood (fwood@stat.columbia.edu).
Implements collapsed Gibbs sampling for the infinite Gaussian mixture model (IGMM).
http://www.robots.ox.ac.uk/~fwood/assets/pdf/Wood-JNM-Preprint-2008.pdf
"""

import numpy, scipy;
import scipy.stats;
import os;

# We will be taking log(0) = -Inf, so turn off this warning
numpy.seterr(divide='ignore')

class MonteCarlo(object):
    """
    @param maximum_number_of_clusters: the maximum number of clusters, used for speeding up the computation
    """
    def __init__(self, maximum_number_of_clusters=100):
        self._maximum_number_of_clusters = maximum_number_of_clusters;

    """
    @param data: a N-by-D numpy array object, defines N points of D dimension
    @param alpha: the concentration parameter of the dirichlet process
    @param kappa_0: initial kappa_0
    @param nu_0: initial nu_0
    @param mu_0: initial cluster center
    @param lambda_0: initial lambda_0
    """
    def _initialize(self, data, alpha=1., kappa_0=1., nu_0=1., mu_0=None, lambda_0=None):
        self._counter = 0;
        
        self._X = data;
        (self._N, self._D) = self._X.shape;
        
        # initialize every point to one cluster
        self._K = 1;
        self._count = numpy.zeros(self._maximum_number_of_clusters, numpy.uint8);
        self._count[0] = self._N;
        self._label = numpy.zeros(self._N, numpy.uint8);
        
        # compute the sum and square sum of all cluster up to truncation level
        self._sum = numpy.zeros((self._maximum_number_of_clusters, self._D));
        self._sum[0, :] = numpy.sum(self._X, 0);
        self._square_sum = numpy.zeros((self._maximum_number_of_clusters, self._D, self._D));
        self._square_sum[[0], :, :] = numpy.dot(self._X.transpose(), self._X);
        
        # initialize the initial mean for the cluster
        if mu_0 == None:
            self._mu_0 = numpy.zeros((1, self._D));
        else:
            self._mu_0 = mu_0;
        assert(self._mu_0.shape == (1, self._D));
        
        # initialize the concentration parameter of the dirichlet distirbution
        self._alpha = alpha;

        # initialize the mean fraction
        self._kappa_0 = kappa_0;
        
        # initialize the degree of freedom
        self._nu_0 = nu_0;
        if self._nu_0 < self._D:
            print "warning: nu_0 is less than data dimensionality, will set to dimensionality..."
            self._nu_0 = self._D;
        
        # initialize the lambda
        if lambda_0 == None:
            self._lambda_0 = numpy.eye(self._D);
        else:
            self._lambda_0 = lambda_0;
        assert(self._lambda_0.shape == (self._D, self._D));

        # initialize the sigma, inv(sigma) and log(det(sigma)) of all cluster up to truncation level
        #self._sigma = numpy.zeros((self._maximum_number_of_clusters, self._D, self._D));
        self._sigma_inv = numpy.zeros((self._maximum_number_of_clusters, self._D, self._D));
        self._log_sigma_det = numpy.zeros(self._maximum_number_of_clusters);
        
        # compute the sigma, inv(sigma) and log(det(sigma)) of the first cluster
        nu = self._nu_0 + self._count[0] - self._D + 1;
        mu_x = self._sum[[0], :] / self._count[0];
        assert(numpy.dot(mu_x.transpose(), mu_x).shape == (self._D, self._D));
        S = self._square_sum[0, :, :] - self._count[0] * numpy.dot(mu_x.transpose(), mu_x);
        mu_x_mu_0 = mu_x - self._mu_0;
        assert(numpy.dot(mu_x_mu_0.transpose(), mu_x_mu_0).shape == (self._D, self._D));
        lambda_n = self._lambda_0 + S + self._kappa_0 * self._count[0] * numpy.dot(mu_x_mu_0.transpose(), mu_x_mu_0) / (self._kappa_0 + self._count[0]);
        sigma = (lambda_n * (self._kappa_0 + self._count[0] + 1)) / ((self._kappa_0 + self._count[0]) * nu);
        assert(numpy.linalg.det(sigma) > 0);
        
        #self._sigma[0, :, :] = sigma;
        self._sigma_inv[0, :, :] = numpy.linalg.inv(sigma);
        self._log_sigma_det[0] = numpy.log(numpy.linalg.det(sigma));
        
        # initialize the default log(det(sigma)) and inv(sigma) for new cluster
        nu = self._nu_0 - self._D + 1;
        sigma_0 = self._lambda_0 * (self._kappa_0 + 1) / (self._kappa_0 * nu);
        self._log_sigma_det_0 = numpy.log(numpy.linalg.det(sigma_0));
        self._sigma_inv_0 = numpy.linalg.inv(sigma_0);
        
    """
    sample the data to train the parameters
    @param iteration: the number of gibbs sampling iteration
    @param directory: the directory to save output, default to "../../output/tmp-output"  
    """
    def learning(self):
        self._counter += 1;

        order = numpy.random.permutation(self._N);
        for (point_counter, point_index) in enumerate(order):
            # get the old label of current point
            old_label = self._label[point_index];
            assert(old_label < self._K and old_label >= 0 and numpy.dtype(old_label) == numpy.uint8);

            # record down the inv(sigma) and log(det(sigma)) of the old cluster
            old_sigma_inv = self._sigma_inv[old_label, :, :];
            old_log_sigma_det = self._log_sigma_det[old_label];

            # remove the current point from the cluster                
            self._count[old_label] -= 1;
            if self._count[old_label] == 0:
                # if current point is from a singleton cluster, shift the last cluster to current one
                self._count[old_label] = self._count[self._K - 1];
                self._label[numpy.nonzero(self._label == (self._K - 1))] = old_label;
                self._sum[old_label, :] = self._sum[self._K - 1, :];
                self._square_sum[old_label, :, :] = self._square_sum[self._K - 1, :, :];
                self._K -= 1;
                # remove the last one to remain compact cluster
                self._count[self._K] = 0;
                self._sum[[self._K], :] = numpy.zeros((1, self._D));
                self._square_sum[[self._K], :, :] = numpy.zeros((1, self._D, self._D));
            else:
                # change the sum and square sum of the old cluster
                self._sum[old_label, :] -= self._X[point_index, :];
                self._square_sum[old_label, :, :] -= numpy.dot(self._X[[point_index], :].transpose(), self._X[[point_index], :]);
                
                # change the inv(sigma) and log(det(sigma)) of the old cluster
                mu_y = self._sum[[old_label], :] / self._count[old_label];
                kappa_n = self._kappa_0 + self._count[old_label];
                nu = self._nu_0 + self._count[old_label] - self._D + 1;
                assert(numpy.dot(mu_y.transpose(), mu_y).shape == (self._D, self._D));
                S = self._square_sum[old_label, :, :] - self._count[old_label] * numpy.dot(mu_y.transpose(), mu_y);
                mu_y_mu_0 = mu_y - self._mu_0;
                assert(numpy.dot(mu_y_mu_0.transpose(), mu_y_mu_0).shape == (self._D, self._D));
                lambda_n = self._lambda_0 + S + self._kappa_0 * self._count[old_label] * numpy.dot(mu_y_mu_0.transpose(), mu_y_mu_0) / (self._kappa_0 + self._count[old_label]);
                sigma = (lambda_n * (kappa_n + 1)) / (kappa_n * nu);
                assert numpy.linalg.det(sigma) > 0, numpy.linalg.det(sigma);

                #self._sigma[old_label, :, :] = sigma;
                self._sigma_inv[old_label, :, :] = numpy.linalg.inv(sigma);
                self._log_sigma_det[old_label] = numpy.linalg.det(sigma);
                
            # compute the prior of being in any of the clusters
            cluster_prior = numpy.hstack((self._count[:self._K], self._alpha));
            cluster_prior = cluster_prior / (self._N - 1. + self._alpha);
            
            # initialize the likelihood vector for all clusters
            cluster_likelihood = numpy.zeros(self._K + 1);
            # compute the likelihood for the existing clusters
            for k in xrange(self._K):
                mu_y = self._sum[[k], :] / self._count[k];
                nu = self._nu_0 + self._count[k] - self._D + 1;
                mu_n = (self._kappa_0 * self._mu_0 + self._count[k] * mu_y) / (self._kappa_0 + self._count[k]);
                y_mu_n = self._X[[point_index], :] - mu_n;
                log_prob = scipy.special.gammaln((nu + self._D) / 2.);
                log_prob -= (scipy.special.gammaln(nu / 2.) + (self._D / 2.) * numpy.log(nu) + (self._D / 2.) * numpy.log(numpy.pi));
                log_prob -= 0.5 * self._log_sigma_det[k];
                assert(numpy.dot(y_mu_n, self._sigma_inv[k, :, :]).shape == (1, self._D));
                assert(numpy.dot(numpy.dot(y_mu_n, self._sigma_inv[k, :, :]), y_mu_n.transpose()).shape == (1, 1))
                log_prob -= 0.5 * (nu + self._D) * numpy.log(1. + 1. / nu * numpy.dot(numpy.dot(y_mu_n, self._sigma_inv[k, :, :]), y_mu_n.transpose()));
                cluster_likelihood[k] = numpy.exp(log_prob);
                
            # compute the likelihood for new cluster
            nu = self._nu_0 - self._D + 1;
            y_mu_0 = self._X[[point_index], :] - self._mu_0;
            log_prob = scipy.special.gammaln((nu + self._D) / 2.);
            log_prob -= (scipy.special.gammaln(nu / 2.) + (self._D / 2.) * numpy.log(nu) + (self._D / 2.) * numpy.log(numpy.pi));
            log_prob -= 0.5 * self._log_sigma_det_0;
            assert(numpy.dot(y_mu_0, self._sigma_inv_0).shape == (1, self._D));
            assert(numpy.dot(numpy.dot(y_mu_0, self._sigma_inv_0), y_mu_0.transpose()).shape == (1, 1));
            log_prob -= 0.5 * (nu + self._D) * numpy.log(1. + 1. / nu * numpy.dot(numpy.dot(y_mu_0, self._sigma_inv_0), y_mu_0.transpose()));
            cluster_likelihood[self._K] = numpy.exp(log_prob);
            
            cluster_posterior = cluster_prior * cluster_likelihood;
            cluster_posterior /= numpy.sum(cluster_posterior);

            # sample a new cluster label for current point                
            cdf = numpy.cumsum(cluster_posterior);
            new_label = numpy.uint8(numpy.nonzero(cdf >= numpy.random.random())[0][0]);
            assert(new_label >= 0 and new_label <= self._K and numpy.dtype(new_label) == numpy.uint8);
            
            # if this point starts up a new cluster
            if new_label == self._K:
                self._K += 1;
            self._count[new_label] += 1;
            self._sum[new_label, :] += self._X[point_index, :];
            self._square_sum[new_label, :, :] += numpy.dot(self._X[[point_index], :].transpose(), self._X[[point_index], :]);
            self._label[point_index] = new_label;
            
            if old_label == new_label:
                # if the point is allocated to the old cluster, retrieve all previous parameter
                self._sigma_inv[new_label, :, :] = old_sigma_inv;
                self._log_sigma_det[new_label] = old_log_sigma_det;
            else:
                # if the point is allocated to a new cluster, compute all new parameter
                mu_y = self._sum[[new_label], :] / self._count[new_label];
                kappa_n = self._kappa_0 + self._count[new_label];
                nu = self._nu_0 + self._count[new_label] - self._D + 1;
                assert(numpy.dot(mu_y.transpose(), mu_y).shape == (self._D, self._D));
                S = self._square_sum[new_label, :, :] - self._count[new_label] * numpy.dot(mu_y.transpose(), mu_y);
                mu_y_mu_0 = mu_y - self._mu_0;
                assert(numpy.dot(mu_y_mu_0.transpose(), mu_y_mu_0).shape == (self._D, self._D));
                lambda_n = self._lambda_0 + S + self._kappa_0 * self._count[new_label] * numpy.dot(mu_y_mu_0.transpose(), mu_y_mu_0) / (self._kappa_0 + self._count[new_label]);
                sigma = (lambda_n * (kappa_n + 1)) / (kappa_n * nu);
                assert(numpy.linalg.det(sigma) > 0);

                #self._sigma[new_label, :, :] = sigma;
                self._sigma_inv[new_label, :, :] = numpy.linalg.inv(sigma);
                self._log_sigma_det[new_label] = numpy.linalg.det(sigma);
        
        return self.log_likelihood();

    """
    """
    def log_likelihood(self):
        log_likelihood = 0.;
        for n in xrange(self._N):
            log_likelihood -= 0.5 * self._D * numpy.log(2.0 * numpy.pi) + 0.5 * self._log_sigma_det[self._label[n]];
            mean_offset = self._X[n, :][numpy.newaxis, :] - self._sum[self._label[n], :][numpy.newaxis, :]/self._count[self._label[n]];
            assert(mean_offset.shape==(1, self._D));
            log_likelihood -= 0.5 * numpy.dot(numpy.dot(mean_offset, self._sigma_inv[self._label[n], :, :]), mean_offset.transpose());

        #@TODO: add in the likelihood for K and hyperparameter
        
        return log_likelihood;
                
    """
    """
    def export_snapshot(self, directory):
        numpy.savetxt(os.path.join(directory, "Label-" + str(self._counter)), numpy.uint8(self._label));
        numpy.savetxt(os.path.join(directory, "Mu-"+ str(self._counter)), self._sum[:self._K, :] / self._count[:self._K][numpy.newaxis, :].transpose());
        sigma = self._sigma_inv;
        for k in xrange(self._K):
            sigma[k, :, :] = numpy.linalg.inv(sigma[k, :, :]);
        numpy.savetxt(os.path.join(directory, "Sigma-" + str(self._counter)), numpy.reshape(sigma[:self._K, :, :], (self._K, self._D * self._D)));
        vector = numpy.array([self._alpha, self._kappa_0, self._nu_0]);
        numpy.savetxt(os.path.join(directory, "Hyperparameter-vector-" + str(self._counter)), vector);
        matrix = numpy.vstack((self._mu_0, self._lambda_0));
        numpy.savetxt(os.path.join(directory, "Hyperparameter-matrix-" + str(self._counter)), matrix);
        
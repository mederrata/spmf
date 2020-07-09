import tensorflow as tf
import tensorflow_probability as tfp

from mederrata_spmf.metastrings import (
    weight_code, cauchy_code, sq_igamma_code, igamma_code)

from mederrata_spmf.util import (
    clip_gradients, run_chain,
    build_trainable_InverseGamma_dist,
    build_trainable_normal_dist
)
from mederrata_spmf.distributions import SqrtInverseGamma
from mederrata_spmf.model import BayesianModel

tfd = tfp.distributions


class Dense(object):
    fn = None
    weights = None
    dtype = tf.float32

    def __init__(
            self, input_size=None, layer_sizes=None,
            activation_fn=None, dtype=tf.float32):

        self.dtype = dtype
        self.activation_fn = tf.nn.relu if (
            activation_fn is None) else activation_fn
        if (input_size is None) or (layer_sizes is None):
            self.fn = lambda x: x
        else:
            self.weights = self.sample_initial_nn_params(
                input_size, layer_sizes)
            self.fn = self.build_network(self.weights)

    def dense(self, X, W, b, activation):
        return activation(tf.matmul(X, W) + b[..., tf.newaxis, :])

    def set_weights(self, weights):
        self.weights = weights
        self.fn = self.build_network(self.weights)

    def build_network(self, weight_tensors, activation=None):
        activation = self.activation_fn if (
            activation is None) else activation

        # @tf.function
        def model(X):
            net = X
            net = tf.cast(net, self.dtype)
            weights_list = weight_tensors[::2]
            biases_list = weight_tensors[1::2]

            for (weights, biases) in zip(weights_list, biases_list):
                net = self.dense(net, weights, biases, activation)
            return net
        return model

    def sample_initial_nn_params(self, input_size, layer_sizes, priors=None):
        """
        Priors should be either none or a list of tuples:
        [(weight prior, bias prior) for layer in layer_sizes]
        """
        architecture = []
        layer_sizes = [input_size] + layer_sizes

        if priors is None:
            for j, layer_size in enumerate(layer_sizes[1:]):
                weights = tfd.Normal(
                    loc=tf.zeros(
                        (layer_sizes[j], layer_size),
                        dtype=self.dtype), scale=1e-1
                ).sample()
                biases = tfd.Normal(
                    loc=tf.zeros((layer_size), dtype=self.dtype), scale=1.
                ).sample()
                architecture += [weights, biases]
        else:
            pass

        return architecture


class DenseHorseshoe(Dense, BayesianModel):
    """Dense horseshoe network of given layer sizes

    Arguments:
        DenseNetwork {[type]} -- [description]
    """
    distribution = None
    surrogate_distribution = None
    reparameterized = True

    def __init__(self, input_size=None, layer_sizes=None,
                 reparameterized=False, decay=0.5,
                 activation_fn=None, dtype=tf.float32, *args, **kwargs):
        super(DenseHorseshoe, self).__init__(
            input_size, layer_sizes,
            activation_fn=activation_fn, dtype=dtype,
            *args, **kwargs)
        self.dtype = dtype
        self.layer_sizes = [input_size] + layer_sizes
        self.reparameterized = reparameterized
        self.assemble_distributions()
        self.decay = decay  # dimensional decay

    def set_weights(self, weights):
        super(DenseHorseshoe, self).set_weights(weights)
        self.assemble_distributions()

    def log_prob(self, x):
        return self.distribution.log_prob(x)

    def sample_weights(self, *args, **kwargs):
        return self.distribution.sample(*args, **kwargs)

    def assemble_networks(self, sample, activation=tf.nn.relu):
        weight_tensors = []
        for j in range(int(len(self.weights)/2)):
            weight_tensors += [sample["w_"+str(j)]] + [sample["b_"+str(j)]]
        net = self.build_network(weight_tensors, activation=activation)
        return net

    def assemble_distributions(self):
        distribution_dict = {}
        factorized_dict = {}
        bijectors = {}
        var_list = []
        weight_var_list = []
        for j, weight in enumerate(self.weights[::2]):
            var_list += ['w_' + str(j)] + ['b_' + str(j)]
            weight_var_list += ['w_' + str(j)] + ['b_' + str(j)]
            model_code = weight_code.format(
                'w',
                j,
                2,
                (self.layer_sizes[j],
                 self.layer_sizes[j+1]),
                "tf." + self.dtype.name)
            distribution_dict['w_' + str(j)] = eval(
                model_code)
            factorized_dict['w_' + str(j)] = build_trainable_normal_dist(
                tf.zeros(
                    (self.layer_sizes[j], self.layer_sizes[j+1]),
                    dtype=self.dtype),
                1e-2 *
                tf.zeros(
                    (self.layer_sizes[j], self.layer_sizes[j+1]),
                    dtype=self.dtype),
                2
            )
            distribution_dict['b_' + str(j)] = eval(
                weight_code.format(
                    'b',
                    j,
                    1,
                    (self.layer_sizes[j+1],),
                    "tf." + self.dtype.name)
            )
            factorized_dict['b_' + str(j)] = build_trainable_normal_dist(
                tf.zeros((self.layer_sizes[j+1],), dtype=self.dtype),
                1e-2*tf.zeros((self.layer_sizes[j+1],), dtype=self.dtype),
                1
            )
            bijectors['w_' + str(j)] = tfp.bijectors.Identity()
            bijectors['b_' + str(j)] = tfp.bijectors.Identity()

            var_list += (
                ['tau_w_{0}'.format(j)] + ['lambda_w_{0}'.format(j)]
                + ['tau_b_{0}'.format(j)] + ['lambda_b_{0}'.format(j)])
            bijectors['tau_w_{0}'.format(j)] = tfp.bijectors.Softplus()
            bijectors['lambda_w_{0}'.format(j)] = tfp.bijectors.Softplus()
            bijectors['tau_b_{0}'.format(j)] = tfp.bijectors.Softplus()
            bijectors['lambda_b_{0}'.format(j)] = tfp.bijectors.Softplus()
            if not self.reparameterized:
                distribution_dict[
                    'tau_w_{0}'.format(j)
                ] = eval(
                    cauchy_code.format(
                        (1, self.layer_sizes[j+1]),
                        1.,
                        2,
                        "tf." + self.dtype.name
                    )
                )
                factorized_dict[
                    'tau_w_{0}'.format(j)
                ] = bijectors['tau_w_{0}'.format(j)](
                    build_trainable_InverseGamma_dist(
                        0.5 *
                        tf.ones((1, self.layer_sizes[j+1]), dtype=self.dtype),
                        tf.ones(
                            (1, self.layer_sizes[j+1]),
                            dtype=self.dtype),
                        1
                    )
                )
                distribution_dict[
                    'tau_b_{0}'.format(j)
                ] = eval(
                    cauchy_code.format(
                        1,
                        1.,
                        1,
                        "tf." + self.dtype.name
                    )
                )
                factorized_dict[
                    'tau_b_{0}'.format(j)
                ] = bijectors['tau_b_{0}'.format(j)](
                    build_trainable_InverseGamma_dist(
                        0.5*tf.ones((1), dtype=self.dtype),
                        tf.ones((1), dtype=self.dtype),
                        1
                    )
                )
                distribution_dict[
                    'lambda_b_{0}'.format(j)
                ] = eval(
                    cauchy_code.format(
                        (self.layer_sizes[j+1],),
                        1.,
                        1,
                        "tf." + self.dtype.name
                    )
                )
                factorized_dict[
                    'lambda_b_{0}'.format(j)
                ] = bijectors['lambda_b_{0}'.format(j)](
                    build_trainable_InverseGamma_dist(
                        0.5 *
                        tf.ones((self.layer_sizes[j+1],), dtype=self.dtype),
                        tf.ones((self.layer_sizes[j+1],), dtype=self.dtype),
                        1
                    )
                )
                distribution_dict[
                    'lambda_w_{0}'.format(j)
                ] = eval(
                    cauchy_code.format(
                        (self.layer_sizes[j], self.layer_sizes[j+1]),
                        1.,
                        2,
                        "tf." + self.dtype.name
                    )
                )
                factorized_dict[
                    'lambda_w_{0}'.format(j)
                ] = bijectors['lambda_w_{0}'.format(j)](
                    build_trainable_InverseGamma_dist(
                        0.5*tf.ones(
                            (self.layer_sizes[j], self.layer_sizes[j+1]), dtype=self.dtype),
                        tf.ones(
                            (self.layer_sizes[j], self.layer_sizes[j+1]), dtype=self.dtype),
                        2
                    )
                )
            else:
                var_list += (
                    ['tau_w_{0}_a'.format(j)] + ['lambda_w_{0}_a'.format(j)]
                    + ['tau_b_{0}_a'.format(j)] + ['lambda_b_{0}_a'.format(j)])
                bijectors['tau_w_{0}_a'.format(j)] = tfp.bijectors.Softplus()
                bijectors[
                    'lambda_w_{0}_a'.format(j)
                ] = tfp.bijectors.Softplus()
                bijectors[
                    'tau_b_{0}_a'.format(j)
                ] = tfp.bijectors.Softplus()
                bijectors[
                    'lambda_b_{0}_a'.format(j)
                ] = tfp.bijectors.Softplus()
                distribution_dict[
                    'lambda_b_{0}'.format(j)
                ] = eval(
                    sq_igamma_code.format(
                        (self.layer_sizes[j+1],),
                        'lambda_b_{0}_a'.format(j),
                        1,
                        "tf." + self.dtype.name
                    )
                )
                factorized_dict[
                    'lambda_b_{0}'.format(j)
                ] = bijectors['lambda_b_{0}'.format(j)](
                    build_trainable_InverseGamma_dist(
                        0.5 *
                        tf.ones((self.layer_sizes[j+1],), dtype=self.dtype),
                        tf.ones((self.layer_sizes[j+1],), dtype=self.dtype),
                        1
                    )
                )

                distribution_dict[
                    'lambda_b_{0}_a'.format(j)
                ] = eval(
                    igamma_code.format(
                        (self.layer_sizes[j+1],),
                        1.0,
                        1,
                        "tf." + self.dtype.name
                    )
                )
                factorized_dict[
                    'lambda_b_{0}_a'.format(j)
                ] = bijectors['lambda_b_{0}_a'.format(j)](
                    build_trainable_InverseGamma_dist(
                        0.5 *
                        tf.ones((self.layer_sizes[j+1],), dtype=self.dtype),
                        tf.ones((self.layer_sizes[j+1],), dtype=self.dtype),
                        1
                    )
                )

                distribution_dict[
                    'lambda_w_{0}'.format(j)
                ] = eval(
                    sq_igamma_code.format(
                        (self.layer_sizes[j], self.layer_sizes[j+1]),
                        'lambda_w_{0}_a'.format(j),
                        2,
                        "tf." + self.dtype.name
                    )
                )

                factorized_dict[
                    'lambda_w_{0}'.format(j)
                ] = bijectors['lambda_w_{0}'.format(j)](
                    build_trainable_InverseGamma_dist(
                        0.5 *
                        tf.ones(
                            (self.layer_sizes[j], self.layer_sizes[j+1]),
                            dtype=self.dtype),
                        tf.ones(
                            (self.layer_sizes[j], self.layer_sizes[j+1]),
                            dtype=self.dtype),
                        2
                    )
                )
                distribution_dict[
                    'lambda_w_{0}_a'.format(j)
                ] = eval(
                    igamma_code.format(
                        (self.layer_sizes[j], self.layer_sizes[j+1]),
                        1.,
                        2,
                        "tf." + self.dtype.name
                    )
                )
                factorized_dict[
                    'lambda_w_{0}_a'.format(j)
                ] = bijectors['lambda_w_{0}_a'.format(j)](
                    build_trainable_InverseGamma_dist(
                        0.5*tf.ones(
                            (self.layer_sizes[j], self.layer_sizes[j+1]),
                            dtype=self.dtype
                        ),
                        tf.ones(
                            (self.layer_sizes[j], self.layer_sizes[j+1]),
                            dtype=self.dtype
                        ),
                        2
                    )
                )
                distribution_dict[
                    'tau_w_{0}'.format(j)
                ] = eval(
                    sq_igamma_code.format(
                        (1, self.layer_sizes[j+1]),
                        'tau_w_{0}_a'.format(j),
                        2,
                        "tf." + self.dtype.name
                    )
                )
                factorized_dict[
                    'tau_w_{0}'.format(j)
                ] = bijectors['tau_w_{0}'.format(j)](
                    build_trainable_InverseGamma_dist(
                        0.5 *
                        tf.ones((1, self.layer_sizes[j+1]), dtype=self.dtype),
                        tf.ones((1, self.layer_sizes[j+1]), dtype=self.dtype),
                        1
                    )
                )
                distribution_dict[
                    'tau_w_{0}_a'.format(j)
                ] = eval(
                    igamma_code.format(
                        (1, self.layer_sizes[j+1]),
                        1.,
                        2,
                        "tf." + self.dtype.name
                    )
                )
                factorized_dict[
                    'tau_w_{0}_a'.format(j)
                ] = bijectors['tau_w_{0}_a'.format(j)](
                    build_trainable_InverseGamma_dist(
                        0.5 *
                        tf.ones((1, self.layer_sizes[j+1]), dtype=self.dtype),
                        tf.ones((1, self.layer_sizes[j+1]), dtype=self.dtype),
                        1
                    )
                )
                distribution_dict[
                    'tau_b_{0}'.format(j)
                ] = eval(
                    sq_igamma_code.format(
                        1,
                        'tau_b_{0}_a'.format(j),
                        1,
                        "tf." + self.dtype.name
                    )
                )
                factorized_dict[
                    'tau_b_{0}'.format(j)
                ] = bijectors['tau_b_{0}'.format(j)](
                    build_trainable_InverseGamma_dist(
                        0.5*tf.ones((1), dtype=self.dtype),
                        tf.ones((1), dtype=self.dtype),
                        1
                    )
                )
                distribution_dict[
                    'tau_b_{0}_a'.format(j)
                ] = eval(
                    igamma_code.format(
                        1,
                        1.,
                        1,
                        "tf." + self.dtype.name
                    )
                )
                factorized_dict[
                    'tau_b_{0}_a'.format(j)
                ] = bijectors['tau_b_{0}_a'.format(j)](
                    build_trainable_InverseGamma_dist(
                        0.5*tf.ones((1), dtype=self.dtype),
                        tf.ones((1), dtype=self.dtype),
                        1
                    )
                )
        self.bijectors = bijectors
        self.joint_prior = tfd.JointDistributionNamed(distribution_dict)
        self.surrogate_distribution = tfd.JointDistributionNamed(
            factorized_dict)
        self.surrogate_distribution_dict = factorized_dict
        self.var_list = var_list
        self.weight_var_list = weight_var_list

    def sample(self, *args, **kwargs):
        return self.surrogate_distribution.sample(*args, **kwargs)


def main():
    denseH = DenseHorseshoe(10, [20, 12, 2],
                            reparameterized=True)
    sample = denseH.joint_prior.sample(10)
    prob = denseH.log_prob(sample)
    sample2 = denseH.surrogate_distribution.sample(10)
    prob2 = denseH.log_prob(sample2)
    networks = denseH.assemble_networks(sample)
    networks(tf.ones((1, 2, 10)))
    return


if __name__ == "__main__":
    main()

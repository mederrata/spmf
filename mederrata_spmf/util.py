import functools
import inspect
import uuid

import numpy as np

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops, math_ops, state_ops
from tensorflow.python.tools.inspect_checkpoint import \
    print_tensors_in_checkpoint_file
from tensorflow.python.training import optimizer
from tensorflow_probability.python import util as tfp_util
from tensorflow_probability.python.bijectors import softplus as softplus_lib
from tensorflow_probability.python.distributions.transformed_distribution import \
    TransformedDistribution
from tensorflow_probability.python.internal import (dtype_util, prefer_static,
                                                    tensorshape_util)
from tensorflow.python.data.ops.dataset_ops import BatchDataset
from tensorflow_probability.python.vi import csiszar_divergence

from mederrata_spmf.distributions import SqrtInverseGamma

tfd = tfp.distributions
tfb = tfp.bijectors


def _trace_loss(loss, grads, variables): return loss


def _trace_variables(loss, grads, variables): return loss, variables


def minimize_distributed(
        loss_fn,
        strategy,
        trainable_variables,
        num_epochs=100000,
        max_decay_steps=25,
        abs_tol=1e-4,
        rel_tol=1e-4,
        trace_fn=_trace_loss,
        learning_rate=1.,
        check_every=25,
        decay_rate=0.95,
        checkpoint_name=None,
        max_initialization_steps=1000,
        tf_dataset=None,
        data_input_signature=None,
        processing_fn=None,
        clip_value=5.,
        name='minimize',
        dtype=tf.float64,
        **kwargs):

    checkpoint_name = str(
        uuid.uuid4()) if checkpoint_name is None else checkpoint_name

    with strategy.scope():
        def batch_normalized_loss(data):
            N = tf.shape(tf.nest.flatten(data)[0])[0]
            loss = loss_fn(data=data)
            return loss/tf.cast(N, loss.dtype)

        train_dist_dataset = strategy.experimental_distribute_dataset(
            tf_dataset)
        iterator = iter(train_dist_dataset)

        learning_rate = 1.0 if learning_rate is None else learning_rate

        def learning_rate_schedule_fn(step):
            return learning_rate*decay_rate**step

        decay_step = 0

        optimizer = tf.optimizers.Adam(
            learning_rate=lambda: learning_rate_schedule_fn(decay_step)
        )
        opt = tfa.optimizers.Lookahead(optimizer)

        checkpoint = tf.train.Checkpoint(
            optimizer=opt, **{
                "var_" + str(j): v for j, v in enumerate(trainable_variables)
            })
        manager = tf.train.CheckpointManager(
            checkpoint, f'./.tf_ckpts/{checkpoint_name}/',
            checkpoint_name=checkpoint_name, max_to_keep=3)
        save_path = manager.save()
        @tf.function
        def train_step(data):
            with tf.GradientTape() as tape:
                loss = batch_normalized_loss(data=data)
                gradients = tape.gradient(loss, trainable_variables)
                gradients = tf.nest.pack_sequence_as(
                    gradients,
                    tf.clip_by_global_norm(
                        [
                            tf.where(
                                tf.math.is_finite(t), t, tf.zeros_like(t)
                            ) for t in tf.nest.flatten(gradients)],
                        clip_value
                    )[0]
                )
                opt.apply_gradients(zip(gradients, trainable_variables))
                return loss

    with strategy.scope():

        @tf.function(input_signature=[train_dist_dataset.element_spec])
        def distributed_train_step(data):
            per_replica_losses = strategy.experimental_run_v2(
                train_step, args=(data,))
            return strategy.reduce(
                tf.distribute.ReduceOp.SUM, per_replica_losses,
                axis=None)

        converged = False
        results = []
        losses = []
        avg_losses = [1e10]*3
        deviations = [1e10]*3
        min_loss = 1e10
        min_state = None

        batches_since_checkpoint = 0
        batches_since_plateau = 0
        accepted_batches = 0
        num_resets = 0
        converged = False
        epoch = 1
        save_path = manager.save()
        print(f"Saved an initial checkpoint: {save_path}")
        for epoch in range(1, num_epochs):
            if converged:
                break
            print(f"Epoch: {epoch}")
            total_loss = 0.
            num_batches = 0

            for data in train_dist_dataset:
                total_loss += distributed_train_step(data)
                num_batches += 1
            train_loss = total_loss / num_batches
            losses += [train_loss]

            if epoch % check_every == 0:
                recent_losses = tf.convert_to_tensor(
                    losses[-check_every:])
                avg_loss = tf.reduce_mean(recent_losses).numpy()

                if not np.isfinite(avg_loss):
                    status = "Backtracking"
                    print(status)
                    cp_status = checkpoint.restore(manager.latest_checkpoint)
                    cp_status.assert_consumed()

                    if accepted_batches == 0:
                        if epoch > max_initialization_steps:
                            converged = True
                            decay_step += 1
                            print(
                                "Failed to initialize within" +
                                f" {max_initialization_steps} steps")
                    else:
                        decay_step += 1

                    epoch += 1
                    cp_status.assert_consumed()
                    print(f" new learning rate: {optimizer.lr}")
                    if num_resets >= max_decay_steps:
                        converged = True
                        print(
                            f"We have reset {num_resets} times so quitting"
                        )
                avg_losses += [avg_loss]
                deviation = tf.math.reduce_std(recent_losses).numpy()
                deviations += [deviation]
                rel = deviation/avg_loss
                status = f"Iteration {epoch} -- loss: {losses[-1].numpy()}, "
                status += f"abs_err: {deviation}, rel_err: {rel}"
                print(status)
                """Check for plateau
                """
                if (((
                        avg_losses[-1] > avg_losses[-3]
                ) and (
                        avg_losses[-1] > avg_losses[-2]
                )) or batches_since_checkpoint > 4
                ) and batches_since_plateau > 2:
                    decay_step += 1
                    if num_resets >= max_decay_steps:
                        converged = True
                        print(
                            f"We have reset {num_resets} times so quitting"
                        )
                    else:
                        status = "We are in a loss plateau"
                        status += f" learning rate: {optimizer.lr}"
                        # status += f" loss: {batch_normalized_loss(data=next(iter(tf_dataset)))}"
                        print(status)
                        cp_status = checkpoint.restore(
                            manager.latest_checkpoint)
                        cp_status.assert_consumed()
                        if tf_dataset is None:
                            status = "Restoring from a checkpoint"
                            # status += f"loss: {loss_fn()}"
                        else:
                            status = "Restoring from a checkpoint"
                            # status += f"{batch_normalized_loss(data=next(iter(tf_dataset)))}"
                        print(status)
                        batches_since_checkpoint = 0
                        batches_since_plateau = 0
                        num_resets += 1
                else:
                    if losses[-1] < min_loss:
                        """
                        Save a checkpoint
                        """
                        min_loss = losses[-1]
                        save_path = manager.save()
                        accepted_batches += 1
                        print(f"Saved a checkpoint: {save_path}")
                        batches_since_checkpoint = 0
                    else:
                        batches_since_checkpoint += 1

                    if deviation < abs_tol:
                        print(
                            f"Converged in {epoch} iterations " +
                            "with acceptable absolute tolerance")
                        converged = True
                    elif rel < rel_tol:
                        print(
                            f"Converged in {epoch} iterations with " +
                            "acceptable relative tolerance")
                        converged = True
                    batches_since_plateau += 1
            epoch += 1
            if epoch >= num_epochs:
                print("Terminating because we are out of iterations")
        return losses


def batched_minimize(loss_fn,
                     num_epochs=1000,
                     max_decay_steps=25,
                     abs_tol=1e-4,
                     rel_tol=1e-4,
                     trainable_variables=None,
                     trace_fn=_trace_loss,
                     learning_rate=1.,
                     check_every=25,
                     decay_rate=0.95,
                     checkpoint_name=None,
                     max_initialization_steps=1000,
                     tf_dataset=None,
                     processing_fn=None,
                     name='minimize',
                     clip_value=10.,
                     **kwargs):

    checkpoint_name = str(
        uuid.uuid4()) if checkpoint_name is None else checkpoint_name
    learning_rate = 1.0 if learning_rate is None else learning_rate

    def learning_rate_schedule_fn(step):
        return learning_rate*decay_rate**step

    decay_step = 0

    optimizer = tf.optimizers.Adam(
        learning_rate=lambda: learning_rate_schedule_fn(decay_step)
    )
    # optimizer = tf.optimizers.SGD(
    #    learning_rate=lambda: learning_rate_schedule_fn(decay_step)
    # )
    opt = tfa.optimizers.Lookahead(optimizer)

    # @tf.function
    def batch_normalized_loss(data):
        N = tf.shape(tf.nest.flatten(data)[0])[0]
        loss = loss_fn(data=data)
        return loss/tf.cast(N, loss.dtype)

    with tf.GradientTape(
            watch_accessed_variables=trainable_variables is None) as tape:
        for v in trainable_variables or []:
            tape.watch(v)
        if tf_dataset is not None:
            loss = batch_normalized_loss(
                data=next(iter(tf_dataset)))
        else:
            loss = loss_fn()
    watched_variables = tape.watched_variables()

    checkpoint = tf.train.Checkpoint(
        optimizer=opt, **{
            "var_" + str(j): v for j, v in enumerate(watched_variables)
        })
    manager = tf.train.CheckpointManager(
        checkpoint, f'./.tf_ckpts/{checkpoint_name}',
        checkpoint_name=checkpoint_name, max_to_keep=3)

    @tf.function(autograph=False)
    def train_loop_body(old_result, step, data=None):  # pylint: disable=unused-argument
        """Run a single optimization step."""
        with tf.GradientTape(
                watch_accessed_variables=trainable_variables is None) as tape:
            for v in trainable_variables or []:
                tape.watch(v)
            if data is not None:
                loss = batch_normalized_loss(
                    data=data)
            else:
                loss = loss_fn()
        watched_variables = tape.watched_variables()
        grads = tape.gradient(loss, watched_variables)
        """
        grads = tf.nest.pack_sequence_as(
            grads,
            [
                tf.clip_by_value(
                    t, -clip_value, clip_value) for t in tf.nest.flatten(grads)]
        )
        """
        grads = tf.nest.pack_sequence_as(
            grads,
            tf.clip_by_global_norm(
                [
                    tf.where(
                        tf.math.is_finite(t), t, tf.zeros_like(t)
                    ) for t in tf.nest.flatten(grads)],
                clip_value
            )[0]
        )
        train_op = opt.apply_gradients(zip(grads, watched_variables))
        with tf.control_dependencies([train_op]):
            state = trace_fn(tf.identity(loss),
                             [tf.identity(g) for g in grads],
                             [tf.identity(v) for v in watched_variables])
        return state

    with tf.name_scope(name) as name:
        # Compute the shape of the trace without executing the graph.
        concrete_loop_body = train_loop_body.get_concrete_function(
            tf.TensorSpec([]), tf.TensorSpec([]))  # Inputs ignored.
        if all([tensorshape_util.is_fully_defined(shape)
                for shape in tf.nest.flatten(
                    concrete_loop_body.output_shapes)]):
            state_initializer = tf.nest.map_structure(
                lambda shape, dtype: tf.zeros(shape, dtype=dtype),
                concrete_loop_body.output_shapes,
                concrete_loop_body.output_dtypes)
            initial_trace_step = None
        else:
            state_initializer = concrete_loop_body(
                tf.convert_to_tensor(0.), tf.convert_to_tensor(0.))
            initial_trace_step = state_initializer

        converged = False
        results = []
        losses = []
        avg_losses = [1e10]*3
        deviations = [1e10]*3
        min_loss = 1e10
        min_state = None
        # Test the first step, and make sure we can initialize safely
        if tf_dataset is not None:
            assert isinstance(tf_dataset, tf.data.Dataset)
            data = next(iter(tf_dataset))
            if processing_fn is not None:
                data = processing_fn(data)
            loss = batch_normalized_loss(data=data)
        else:
            loss = loss_fn()
        if not np.isfinite(loss.numpy()):
            # print(loss)
            print("Failed to initialize")
            converged = True
        else:
            print(f"Initial loss: {loss}")

        step = tf.cast(1, tf.int32)
        batches_since_checkpoint = 0
        batches_since_plateau = 0
        accepted_batches = 0
        num_resets = 0
        while (step < num_epochs) and not converged:
            if tf_dataset is None:
                losses += [
                    train_loop_body(state_initializer, step)
                ]
            else:
                batch_losses = []
                for data in tf_dataset:
                    if processing_fn is not None:
                        data = processing_fn(data)
                    batch_loss = train_loop_body(
                            state_initializer, step, data
                        )
                    if not np.isfinite(batch_loss.numpy()):
                        cp_status = checkpoint.restore(manager.latest_checkpoint)
                        cp_status.assert_consumed()

                        batch_loss = train_loop_body(
                            state_initializer, step, data
                        )
                        decay_step += 1
                    if np.isfinite(batch_loss.numpy()):
                        batch_losses += [batch_loss]
                    else:
                        print("Batch loss NaN")

            loss = tf.reduce_mean(batch_losses)
            avg_losses += [loss]
            losses += [loss]
            deviation = tf.math.reduce_std(batch_losses)
            rel = deviation/loss
            print(
                f"Epoch {step}: average-batch loss: {loss} last batch loss: {batch_loss}")

            if True:  # step % check_every == 0:

                """Check for convergence
                """
                if not np.isfinite(loss):
                    cp_status = checkpoint.restore(manager.latest_checkpoint)
                    cp_status.assert_consumed()

                    #raise ArithmeticError(
                    #    "We are NaN, restored the last checkpoint")
                    print("Got NaN, restoring a checkpoint")
                    decay_step += 1

                """Check for plateau
                """
                if (((
                        avg_losses[-1] > avg_losses[-3]
                ) and (
                        avg_losses[-1] > avg_losses[-2]
                ))
                ) and batches_since_plateau > 3:
                    decay_step += 1
                    if num_resets >= max_decay_steps:
                        converged = True
                        print(
                            f"We have reset {num_resets} times so quitting"
                        )
                    else:
                        status = "We are in a loss plateau"
                        status += f" learning rate: {optimizer.lr}"
                        status += f" loss: {batch_normalized_loss(data=next(iter(tf_dataset)))}"
                        print(status)
                        # cp_status = checkpoint.restore(
                        #    manager.latest_checkpoint)
                        # cp_status.assert_consumed()
                        if tf_dataset is None:
                            status = "Restoring from a checkpoint - "
                            status += f"loss: {loss_fn()}"
                        else:
                            status = "Restoring from a checkpoint - loss: "
                            status += f"{batch_normalized_loss(data=next(iter(tf_dataset)))}"
                        print(status)
                        batches_since_checkpoint = 0
                        batches_since_plateau = 0
                        num_resets += 1
                else:
                    if losses[-1] < min_loss:
                        """
                        Save a checkpoint
                        """
                        min_loss = losses[-1]
                        save_path = manager.save()
                        accepted_batches += 1
                        print(f"Saved a checkpoint: {save_path}")
                        batches_since_checkpoint = 0
                    else:
                        batches_since_checkpoint += 1

                    if deviation < abs_tol:
                        print(
                            f"Converged in {step} iterations " +
                            "with acceptable absolute tolerance")
                        converged = True
                    elif rel < rel_tol:
                        print(
                            f"Converged in {step} iterations with " +
                            "acceptable relative tolerance")
                        converged = True
                    batches_since_plateau += 1
            step += 1
            if step >= num_epochs:
                print("Terminating because we are out of iterations")

        trace = tf.stack(losses)
        if initial_trace_step is not None:
            trace = tf.nest.map_structure(
                lambda a, b: tf.concat([a[tf.newaxis, ...], b], axis=0),
                initial_trace_step, trace)
        cp_status = checkpoint.restore(manager.latest_checkpoint)
        cp_status.assert_consumed()
        return trace


def clip_gradients(fn, clip_value, dtype=tf.float64):
    def wrapper(*args, **kwargs):
        @tf.custom_gradient
        def grad_wrapper(*flat_args_kwargs):
            with tf.GradientTape() as tape:
                tape.watch(flat_args_kwargs)
                new_args, new_kwargs = tf.nest.pack_sequence_as(
                    (args, kwargs),
                    flat_args_kwargs)
                ret = fn(*new_args, **new_kwargs)

            def grad_fn(*dy):
                flat_grads = tape.gradient(
                    ret, flat_args_kwargs, output_gradients=dy)
                flat_grads = tf.nest.map_structure(
                    lambda g: tf.where(tf.math.is_finite(g),
                                       g, tf.zeros_like(g)),
                    flat_grads)
                return tf.clip_by_global_norm(flat_grads, clip_value)[0]
            return ret, grad_fn
        return grad_wrapper(*[tf.nest.flatten((args, kwargs))])
    return wrapper


@tf.function
def run_chain(
        init_state, step_size, target_log_prob_fn,
        unconstraining_bijectors, num_steps=500,
        burnin=50, num_leapfrog_steps=5, nuts=True
):
    if nuts:
        def trace_fn(_, pkr):
            return (
                pkr.inner_results.inner_results.target_log_prob,
                pkr.inner_results.inner_results.leapfrogs_taken,
                pkr.inner_results.inner_results.has_divergence,
                pkr.inner_results.inner_results.energy,
                pkr.inner_results.inner_results.log_accept_ratio
            )

        kernel = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel=tfp.mcmc.NoUTurnSampler(
                target_log_prob_fn,
                step_size=step_size),
            bijector=unconstraining_bijectors)

        hmc = tfp.mcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=kernel,
            num_adaptation_steps=burnin,
            step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(
                inner_results=pkr.inner_results._replace(
                    step_size=new_step_size)),
            step_size_getter_fn=lambda pkr: pkr.inner_results.step_size,
            log_accept_prob_getter_fn=lambda pkr: (
                pkr.inner_results.log_accept_ratio)
        )

        # Sampling from the chain.
        chain_state, sampler_stat = tfp.mcmc.sample_chain(
            num_results=num_steps,
            num_burnin_steps=burnin,
            current_state=init_state,
            kernel=hmc,
            trace_fn=trace_fn)
    else:
        def trace_fn_hmc(_, pkr):
            return (pkr.inner_results.inner_results.is_accepted,
                    pkr.inner_results.inner_results.accepted_results.step_size)
        hmc = tfp.mcmc.TransformedTransitionKernel(
            inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=target_log_prob_fn,
                num_leapfrog_steps=num_leapfrog_steps,
                step_size=step_size,
                state_gradients_are_stopped=True),
            bijector=unconstraining_bijectors)
        kernel = tfp.mcmc.SimpleStepSizeAdaptation(
            inner_kernel=hmc, num_adaptation_steps=int(0.8*burnin))
        chain_state, sampler_stat = tfp.mcmc.sample_chain(
            num_results=num_steps,
            num_burnin_steps=burnin,
            current_state=init_state,
            kernel=kernel,
            trace_fn=trace_fn_hmc)

    return chain_state, sampler_stat


def build_trainable_concentration_scale_distribution(
        initial_concentration,
        initial_scale,
        event_ndims,
        distribution_fn=tfd.InverseGamma,
        validate_args=False,
        strategy=None,
        name=None):
    """Builds a variational distribution from a location-scale family.
    Args:
      initial_concentration: Float `Tensor` initial concentration.
      initial_scale: Float `Tensor` initial scale.
      event_ndims: Integer `Tensor` number of event dimensions
        in `initial_concentration`.
      distribution_fn: Optional constructor for a `tfd.Distribution` instance
        in a location-scale family. This should have signature `dist =
        distribution_fn(loc, scale, validate_args)`.
        Default value: `tfd.Normal`.
      validate_args: Python `bool`. Whether to validate input with asserts.
        This imposes a runtime cost. If `validate_args` is `False`, and the
        inputs are invalid, correct behavior is not guaranteed.
        Default value: `False`.
      name: Python `str` name prefixed to ops created by this function.
        Default value: `None` (i.e.,
          'build_trainable_location_scale_distribution').
    Returns:
      posterior_dist: A `tfd.Distribution` instance.
    """
    scope = strategy.scope() if strategy is not None else tf.name_scope(
        name or 'build_trainable_concentration_scale_distribution')
    with scope:
        dtype = dtype_util.common_dtype([initial_concentration, initial_scale],
                                        dtype_hint=tf.float64)
        initial_concentration = tf.cast(
            initial_concentration, dtype=dtype)
        initial_scale = tf.cast(initial_scale, dtype=dtype)

        loc = TransformedVariable(
            initial_concentration,
            softplus_lib.Softplus(),
            scope=scope,
            name='concentration')
        scale = TransformedVariable(
            initial_scale, softplus_lib.Softplus(),
            scope=scope, name='scale')
        posterior_dist = distribution_fn(concentration=loc, scale=scale,
                                         validate_args=validate_args)

        # Ensure the distribution has the desired number of event dimensions.
        static_event_ndims = tf.get_static_value(event_ndims)
        if static_event_ndims is None or static_event_ndims > 0:
            posterior_dist = tfd.Independent(
                posterior_dist,
                reinterpreted_batch_ndims=event_ndims,
                validate_args=validate_args)

    return posterior_dist


def build_trainable_location_scale_distribution(initial_loc,
                                                initial_scale,
                                                event_ndims,
                                                distribution_fn=tfd.Normal,
                                                validate_args=False,
                                                strategy=None,
                                                name=None):
    """Builds a variational distribution from a location-scale family.
    Args:
      initial_loc: Float `Tensor` initial location.
      initial_scale: Float `Tensor` initial scale.
      event_ndims: Integer `Tensor` number of event dimensions
                    in `initial_loc`.
      distribution_fn: Optional constructor for a `tfd.Distribution` instance
        in a location-scale family. This should have signature `dist =
        distribution_fn(loc, scale, validate_args)`.
        Default value: `tfd.Normal`.
      validate_args: Python `bool`. Whether to validate input with asserts.
        This
        imposes a runtime cost. If `validate_args` is `False`, and the inputs are
        invalid, correct behavior is not guaranteed.
        Default value: `False`.
      name: Python `str` name prefixed to ops created by this function.
        Default value: `None` (i.e.,
          'build_trainable_location_scale_distribution').
    Returns:
      posterior_dist: A `tfd.Distribution` instance.
    """
    scope = strategy.scope() if strategy is not None else tf.name_scope(
        name or 'build_trainable_location_scale_distribution')
    with scope:
        dtype = dtype_util.common_dtype([initial_loc, initial_scale],
                                        dtype_hint=tf.float32)
        initial_loc = tf.convert_to_tensor(initial_loc, dtype=dtype)
        initial_scale = tf.convert_to_tensor(initial_scale, dtype=dtype)

        loc = tf.Variable(initial_value=initial_loc, name='loc')
        scale = TransformedVariable(
            initial_scale, softplus_lib.Softplus(), scope=scope,
            name='scale')
        posterior_dist = distribution_fn(loc=loc, scale=scale,
                                         validate_args=validate_args)

        # Ensure the distribution has the desired number of event dimensions.
        static_event_ndims = tf.get_static_value(event_ndims)
        if static_event_ndims is None or static_event_ndims > 0:
            posterior_dist = tfd.Independent(
                posterior_dist,
                reinterpreted_batch_ndims=event_ndims,
                validate_args=validate_args)

    return posterior_dist


class TransformedVariable(tfp_util.TransformedVariable):
    def __init__(self, initial_value, bijector,
                 dtype=None, scope=None, name=None, **kwargs):
        """Creates the `TransformedVariable` object.

        Args:
        initial_value: A `Tensor`, or Python object convertible to a `Tensor`,
            which is the initial value for the Variable. Can also be a callable with
            no argument that returns the initial value when called. Note: if
            `initial_value` is a `TransformedVariable` then the instantiated object
            does not create a new `tf.Variable`, but rather points to the underlying
            `Variable` and chains the `bijector` arg with the underlying bijector as
            `tfb.Chain([bijector, initial_value.bijector])`.
        bijector: A `Bijector`-like instance which defines the transformations
            applied to the underlying `tf.Variable`.
        dtype: `tf.dtype.DType` instance or otherwise valid `dtype` value to
            `tf.convert_to_tensor(..., dtype)`.
            Default value: `None` (i.e., `bijector.dtype`).
        name: Python `str` representing the underlying `tf.Variable`'s name.
            Default value: `None`.
        **kwargs: Keyword arguments forward to `tf.Variable`.
        """
        # Check if `bijector` is "`Bijector`-like".
        for attr in {'forward', 'forward_event_shape',
                     'inverse', 'inverse_event_shape',
                     'name', 'dtype'}:
            if not hasattr(bijector, attr):
                raise TypeError('Argument `bijector` missing required `Bijector` '
                                'attribute "{}".'.format(attr))

        if callable(initial_value):
            initial_value = initial_value()
        initial_value = tf.convert_to_tensor(
            initial_value, dtype_hint=bijector.dtype, dtype=dtype)

        if scope is not None:
            with scope:
                variable = tf.Variable(
                    initial_value=bijector.inverse(initial_value),
                    name=name,
                    dtype=dtype,
                    **kwargs)
        else:
            variable = tf.Variable(
                initial_value=bijector.inverse(initial_value),
                name=name,
                dtype=dtype,
                **kwargs)
        super(tfp_util.TransformedVariable, self).__init__(
            pretransformed_input=variable,
            transform_fn=bijector,
            shape=initial_value.shape,
            name=bijector.name)
        self._bijector = bijector


build_trainable_InverseGamma_dist = functools.partial(
    build_trainable_concentration_scale_distribution,
    distribution_fn=tfd.InverseGamma
)

build_trainable_normal_dist = functools.partial(
    build_trainable_location_scale_distribution,
    distribution_fn=tfd.Normal)

_reparameterized_elbo = functools.partial(
    csiszar_divergence.monte_carlo_variational_loss,
    discrepancy_fn=csiszar_divergence.kl_reverse,
    use_reparameterization=True)


def minibatch_mc_variational_loss(target_log_prob_fn,
                                  surrogate_posterior,
                                  sample_size=1,
                                  discrepancy_fn=tfp.vi.kl_reverse,
                                  use_reparameterization=None,
                                  seed=None,
                                  data=None,
                                  strategy=None,
                                  name=None):
    _target_log_prob_fn = functools.partial(
        target_log_prob_fn, data=data
    )

    return tfp.vi.monte_carlo_variational_loss(
        _target_log_prob_fn,
        surrogate_posterior,
        sample_size=sample_size,
        discrepancy_fn=discrepancy_fn,
        use_reparameterization=use_reparameterization,
        seed=seed,
        name=name
    )


def fit_surrogate_posterior(target_log_prob_fn,
                            surrogate_posterior,
                            num_epochs=1000,
                            trace_fn=_trace_loss,
                            variational_loss_fn=_reparameterized_elbo,
                            sample_size=5,
                            check_every=25,
                            decay_rate=0.9,
                            learning_rate=1.0,
                            max_decay_steps=25,
                            clip_value=10.,
                            trainable_variables=None,
                            seed=None,
                            abs_tol=None,
                            rel_tol=None,
                            tf_dataset=None,
                            strategy=None,
                            name=None,
                            **kwargs):
    if trainable_variables is None:
        trainable_variables = surrogate_posterior.trainable_variables

    def complete_variational_loss_fn(data=None):
        """This becomes the loss function called in the 
        optimization loop

        Keyword Arguments:
            data {tf.data.Datasets batch} -- [description] (default: {None})

        Returns:
            [type] -- [description]
        """
        return minibatch_mc_variational_loss(
            target_log_prob_fn,
            surrogate_posterior,
            sample_size=sample_size,
            data=data,
            seed=seed,
            strategy=strategy,
            name=name,
            **kwargs)
    if strategy is None:
        return batched_minimize(complete_variational_loss_fn,
                                num_epochs=num_epochs,
                                max_decay_steps=max_decay_steps,
                                trace_fn=trace_fn,
                                learning_rate=learning_rate,
                                trainable_variables=trainable_variables,
                                abs_tol=abs_tol,
                                rel_tol=rel_tol,
                                clip_value=clip_value,
                                decay_rate=decay_rate,
                                tf_dataset=tf_dataset,
                                check_every=check_every,
                                **kwargs)
    else:
        return minimize_distributed(
            complete_variational_loss_fn,
            num_epochs=num_epochs,
            max_decay_steps=max_decay_steps,
            trace_fn=trace_fn,
            learning_rate=learning_rate,
            trainable_variables=trainable_variables,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            decay_rate=decay_rate,
            tf_dataset=tf_dataset,
            check_every=check_every,
            strategy=strategy,
            **kwargs)


def build_surrogate_posterior(joint_distribution_named,
                              bijectors=None,
                              exclude=[],
                              num_samples=25,
                              initializers={},
                              strategy=None,
                              name=None,
                              dtype=tf.float64):

    prior_sample = joint_distribution_named.sample(int(num_samples))
    surrogate_dict = {}
    means = {
        k: tf.cast(
            tf.reduce_mean(
                v, axis=0
            ), dtype=dtype
        ) for k, v in prior_sample.items()}

    prior_sample = joint_distribution_named.sample()
    bijectors = defaultdict(tfb.Identity) if bijectors is None else bijectors
    for k, v in joint_distribution_named.model.items():
        if k in exclude:
            continue
        if callable(v):
            test_input = {
                a: prior_sample[a] for a in inspect.getfullargspec(v).args}
            test_distribution = v(**test_input)
        else:
            test_distribution = v
        if isinstance(
            test_distribution.distribution, tfd.InverseGamma
        ) or isinstance(test_distribution.distribution, SqrtInverseGamma):
            surrogate_dict[k] = bijectors[k](
                build_trainable_InverseGamma_dist(
                    2.*tf.ones(test_distribution.event_shape, dtype=dtype),
                    tf.ones(test_distribution.event_shape, dtype=dtype),
                    len(test_distribution.event_shape),
                    strategy=strategy,
                    name=name
                )
            )
        else:
            if k in initializers.keys():
                loc = initializers[k]
            else:
                loc = means[k]
            surrogate_dict[k] = bijectors[k](
                build_trainable_normal_dist(
                    tfb.Invert(bijectors[k])(loc),
                    1e-3*tf.ones(test_distribution.event_shape, dtype=dtype),
                    len(test_distribution.event_shape),
                    strategy=strategy,
                    name=name
                )
            )
    return tfd.JointDistributionNamed(surrogate_dict)

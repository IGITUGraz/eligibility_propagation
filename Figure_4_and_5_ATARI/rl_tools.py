import tensorflow as tf
import collections

from tensorflow.python.util import nest

from util import combine_flat_list, switch_time_and_batch_dimension


ActorOutput = collections.namedtuple(
    'ActorOutput', 'level_name agent_state env_outputs agent_outputs '
                   'action_probabilities')


def create_local_states(agent, envs):
    structure = envs[0].initial()
    flat_list = [[a[None] for a in nest.flatten(env.initial())] for env in envs]
    initial_env_output, initial_env_state = combine_flat_list(structure, flat_list, axis=0)
    n_batch = len(envs)
    initial_agent_state = agent.initial_state(n_batch)
    initial_action = tf.zeros([n_batch], dtype=tf.int64)
    dummy_agent_output, _, dummy_custom_rnn_output, dummy_torso_output = agent(
        (initial_action,
         initial_env_output),
        initial_agent_state)

    initial_agent_output = nest.map_structure(
        lambda t: tf.zeros(t.shape, t.dtype), dummy_agent_output)

    def create_state(t):
        with tf.variable_scope(None, default_name='state'):
            return tf.get_local_variable(t.op.name, initializer=t, use_resource=True)

    persistent_state = nest.map_structure(
        create_state, (initial_env_state, initial_env_output, initial_agent_state,
                       initial_agent_output))
    first_values = nest.map_structure(lambda v: v.read_value(), persistent_state)

    return persistent_state, first_values, structure, dummy_torso_output, dummy_custom_rnn_output


def build_actors(agent, envs, level_name, unroll_length, action_set, observation_structure,
                 first_values, dummy_torso_output, dummy_custom_rnn_output,
                 return_rnn_activity=False):
    def step(input_, _):
        (env_state, env_output, agent_state, agent_output), _, _ = input_

        action = agent_output.action
        batched_env_output = env_output
        agent_output, agent_state, custom_rnn_output, torso_output = \
            agent((action, batched_env_output), agent_state)

        action = agent_output[0]
        raw_action = tf.gather(action_set, action)

        flat_list = []
        for i, env in enumerate(envs):
            env_out = env.step(raw_action[i], nest.map_structure(lambda a: a[i], env_state))
            flat_list.append([a[None] for a in nest.flatten(env_out)])
        env_output, env_state = combine_flat_list(observation_structure, flat_list, axis=0)

        return (env_state, env_output, agent_state, agent_output), torso_output, custom_rnn_output

    first_env_output = first_values[1]
    first_agent_state = first_values[2]
    first_agent_output = first_values[3]

    output = tf.scan(step, tf.range(unroll_length), (first_values, tf.zeros_like(dummy_torso_output),
                                                            nest.map_structure(tf.zeros_like, dummy_custom_rnn_output)))
    (_, env_outputs, _, agent_outputs), torso_outputs, custom_rnn_outputs = output
    if len(torso_outputs.get_shape()) == 4:
        torso_outputs = tf.transpose(torso_outputs, (0, 2, 1, 3))
    rnn_h = custom_rnn_outputs[0]
    rnn_c = custom_rnn_outputs[1]

    full_agent_outputs, full_env_outputs = nest.map_structure(
        lambda first, rest: tf.concat([[first], rest], 0),
        (first_agent_output, first_env_output), (agent_outputs, env_outputs))

    action_probabilities = switch_time_and_batch_dimension(tf.nn.softmax(full_agent_outputs.policy_logits)[1:])

    update_values = (torso_outputs, custom_rnn_outputs, full_env_outputs, output, full_agent_outputs)

    output = ActorOutput(
        level_name=level_name, agent_state=first_agent_state,
        env_outputs=nest.map_structure(switch_time_and_batch_dimension, full_env_outputs),
        agent_outputs=nest.map_structure(switch_time_and_batch_dimension, full_agent_outputs),
        action_probabilities=action_probabilities
    )

    if return_rnn_activity:
        return nest.map_structure(tf.stop_gradient, output), update_values, rnn_h, rnn_c
    return nest.map_structure(tf.stop_gradient, output), update_values


def update_states(update_values, persistent_state):
    torso_outputs, custom_rnn_outputs, env_outputs, output, full_agent_outputs = update_values

    assign_ops = nest.map_structure(lambda v, t: v.assign(t[-1]),
                                    persistent_state, output[0])

    return assign_ops

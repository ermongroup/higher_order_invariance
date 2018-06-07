import numpy as np
import tensorflow as tf
from baselines import logger
import baselines.common as common
from baselines.common import tf_util as U
from baselines.acktr import kfac
from baselines.common.filters import ZFilter
from baselines.acktr.policies import GaussianMlpPolicy
from baselines.common.misc_util import zipsame


def pathlength(path):
    return path["reward"].shape[0]# Loss function that we'll differentiate to get the policy gradient

def rollout(env, policy, max_pathlength, animate=False, obfilter=None):
    """
    Simulate the env and policy for max_pathlength steps
    """
    ob = env.reset()
    prev_ob = np.float32(np.zeros(ob.shape))
    if obfilter: ob = obfilter(ob)
    terminated = False

    obs = []
    acs = []
    ac_dists = []
    logps = []
    rewards = []
    for _ in range(max_pathlength):
        if animate:
            env.render()
        state = np.concatenate([ob, prev_ob], -1)
        obs.append(state)
        ac, ac_dist, logp = policy.act(state)
        acs.append(ac)
        ac_dists.append(ac_dist)
        logps.append(logp)
        prev_ob = np.copy(ob)
        scaled_ac = env.action_space.low + (ac + 1.) * 0.5 * (env.action_space.high - env.action_space.low)
        scaled_ac = np.clip(scaled_ac, env.action_space.low, env.action_space.high)
        ob, rew, done, _ = env.step(scaled_ac)
        if obfilter: ob = obfilter(ob)
        rewards.append(rew)
        if done:
            terminated = True
            break
    return {"observation" : np.array(obs), "terminated" : terminated,
            "reward" : np.array(rewards), "action" : np.array(acs),
            "action_dist": np.array(ac_dists), "logp" : np.array(logps)}

def learn(env, policy, vf, gamma, lam, timesteps_per_batch, num_timesteps,
    animate=False, callback=None, desired_kl=0.002, lr=0.03, momentum=0.9):
    ob_dim, ac_dim = policy.ob_dim, policy.ac_dim
    dbpi = GaussianMlpPolicy(ob_dim, ac_dim, 'dbp')
    oldpi = GaussianMlpPolicy(ob_dim, ac_dim, 'oe')
    dboldpi = GaussianMlpPolicy(ob_dim, ac_dim, 'doi')
    # with tf.variable_scope('dbp'):
    # with tf.variable_scope('oe'):
    # with tf.variable_scope('doi'):

    pi = policy

    do_std = U.function([], [pi.std_1a, pi.logstd_1a])

    kloldnew = oldpi.pd.kl(pi.pd)
    dbkloldnew = dboldpi.pd.kl(dbpi.pd)
    dist = meankl = tf.reduce_mean(kloldnew)
    dbkl = tf.reduce_mean(dbkloldnew)
    obfilter = ZFilter(env.observation_space.shape)

    max_pathlength = env.spec.timestep_limit
    stepsize = tf.Variable(initial_value=np.float32(np.array(lr)), name='stepsize')
    inputs, loss, loss_sampled = policy.update_info

    var_list = [v for v in tf.global_variables() if "pi" in v.name]
    db_var_list = [v for v in tf.global_variables() if "dbp" in v.name]
    old_var_list = [v for v in tf.global_variables() if "oe" in v.name]
    db_old_var_list = [v for v in tf.global_variables() if "doi" in v.name]
    print(len(var_list), len(db_var_list), len(old_var_list), len(db_old_var_list))
    assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                                                    for (oldv, newv) in
                                                    zipsame(old_var_list, var_list)])
    assign_db = U.function([], [], updates=
    [tf.assign(db, o) for (db, o) in zipsame(db_var_list, var_list)] +
    [tf.assign(dbold, dbnew) for (dbold, dbnew) in zipsame(db_old_var_list, old_var_list)])

    assign_old_eq_newr = U.function([], [], updates=[tf.assign(newv, oldv)
                                                    for (oldv, newv) in
                                                    zipsame(old_var_list, var_list)])
    # assign_dbr = U.function([], [], updates=
    # [tf.assign(o, db) for (db, o) in zipsame(db_var_list, var_list)] +
    # [tf.assign(dbnew, dbold) for (dbold, dbnew) in zipsame(db_old_var_list, old_var_list)])

    klgrads = tf.gradients(dist, var_list)
    dbklgrads = tf.gradients(dbkl, db_var_list)
    p_grads = [tf.ones_like(v) for v in dbklgrads]

    get_flat = U.GetFlat(var_list)
    get_old_flat = U.GetFlat(old_var_list)
    set_from_flat = U.SetFromFlat(var_list)

    flat_tangent2 = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan2")
    shapes = [var.get_shape().as_list() for var in var_list]
    start = 0
    tangents2 = []
    for shape in shapes:
        sz = U.intprod(shape)
        tangents2.append(tf.reshape(flat_tangent2[start:start + sz], shape))
        start += sz
    gvp2 = tf.add_n([tf.reduce_sum(g * tangent2) for (g, tangent2) in zipsame(dbklgrads, tangents2)])
    gvp2_grads = tf.gradients(gvp2, db_var_list)

    neg_term = tf.add_n([tf.reduce_sum(g * tangent2) for (g, tangent2) in zipsame(gvp2_grads, tangents2)]) / 2.
    ng1 = tf.gradients(neg_term, db_var_list)
    ng2 = tf.gradients(neg_term, db_old_var_list) 

    neg_term_grads = [a + b for (a, b) in
                      zip(tf.gradients(neg_term, db_var_list), tf.gradients(neg_term, db_old_var_list))]
    neg_term = neg_term_grads
    # neg_term = tf.concat(axis=0, values=[tf.reshape(v, [U.numel(v)]) for v in neg_term_grads])

    pos_term = tf.add_n([tf.reduce_sum(g * tangent) for (g, tangent) in zipsame(gvp2_grads, p_grads)])
    pos_term_grads = [a + b for (a, b) in
                      zip(tf.gradients(pos_term, db_var_list), tf.gradients(pos_term, db_old_var_list))]
    pos_term_sum = tf.add_n([tf.reduce_sum(g * tangent) for (g, tangent) in zipsame(pos_term_grads, tangents2)])
    pos_term_grads = tf.gradients(pos_term_sum, p_grads)
    pos_term = pos_term_grads
    # pos_term = tf.concat(axis=0, values=[tf.reshape(v, [U.numel(v)]) for v in pos_term_grads])
    geo_term = [(p - n) * 0.5 for p, n in zip(pos_term, neg_term)]

    optim = kfac.KfacOptimizer(learning_rate=stepsize, cold_lr=stepsize*(1-0.9), momentum=momentum, kfac_update=2,\
                                epsilon=1e-2, stats_decay=0.99, async=1, cold_iter=1,
                                weight_decay_dict=policy.wd_dict, max_grad_norm=None)
    pi_var_list = []
    for var in tf.trainable_variables():
        if "pi" in var.name:
            pi_var_list.append(var)

    grads = optim.compute_gradients(loss, var_list=pi_var_list)
    update_op, q_runner = optim.minimize(loss, loss_sampled, var_list=pi_var_list)
    geo_term = [g1 + g2[0] for g1, g2 in zip(geo_term, grads)]
    geo_grads = list(zip(geo_term, var_list))
    update_geo_op, q_runner_geo = optim.apply_gradients(geo_grads)
    do_update = U.function(inputs, update_op)
    inputs_tangent = list(inputs) + [flat_tangent2]
    do_update_geo = U.function(inputs_tangent, update_geo_op)
    do_get_geo_term = U.function(inputs_tangent, [ng1, ng2])
    U.initialize()

    # start queue runners
    enqueue_threads = []
    coord = tf.train.Coordinator()
    for qr in [q_runner, vf.q_runner, q_runner_geo]:
        assert (qr != None)
        enqueue_threads.extend(qr.create_threads(tf.get_default_session(), coord=coord, start=True))

    i = 0
    timesteps_so_far = 0
    while True:
        if timesteps_so_far > num_timesteps:
            break
        logger.log("********** Iteration %i ************"%i)

        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            path = rollout(env, policy, max_pathlength, animate=(len(paths)==0 and (i % 10 == 0) and animate), obfilter=obfilter)
            paths.append(path)
            n = pathlength(path)
            timesteps_this_batch += n
            timesteps_so_far += n
            if timesteps_this_batch > timesteps_per_batch:
                break

        # Estimate advantage function
        vtargs = []
        advs = []
        for path in paths:
            rew_t = path["reward"]
            return_t = common.discount(rew_t, gamma)
            vtargs.append(return_t)
            vpred_t = vf.predict(path)
            vpred_t = np.append(vpred_t, 0.0 if path["terminated"] else vpred_t[-1])
            delta_t = rew_t + gamma*vpred_t[1:] - vpred_t[:-1]
            adv_t = common.discount(delta_t, gamma * lam)
            advs.append(adv_t)
        # Update value function
        vf.fit(paths, vtargs)

        # Build arrays for policy update
        ob_no = np.concatenate([path["observation"] for path in paths])
        action_na = np.concatenate([path["action"] for path in paths])
        oldac_dist = np.concatenate([path["action_dist"] for path in paths])
        adv_n = np.concatenate(advs)
        standardized_adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)

        assign_old_eq_new()  # set old parameter values to new parameter values
        assign_db()

        # Policy update
        do_update(ob_no, action_na, standardized_adv_n)
        # ft2 = get_flat() - get_old_flat()

        # assign_old_eq_newr() # assign back
        # gnp = do_get_geo_term(ob_no, action_na, standardized_adv_n, ft2)

        # def check_nan(bs):
        #     return [~np.isnan(b).all() for b in bs]

        # print(gnp[0])
        # print('.....asdfasdfadslfkadsjfaksdfalsdkfjaldskf')
        # print(gnp[1])
        # do_update_geo(ob_no, action_na, standardized_adv_n, ft2)

        min_stepsize = np.float32(1e-8)
        max_stepsize = np.float32(1e0)
        # Adjust stepsize
        kl = policy.compute_kl(ob_no, oldac_dist)
        # if kl > desired_kl * 2:
        #     logger.log("kl too high")
        #     tf.assign(stepsize, tf.maximum(min_stepsize, stepsize / 1.5)).eval()
        # elif kl < desired_kl / 2:
        #     logger.log("kl too low")
        #     tf.assign(stepsize, tf.minimum(max_stepsize, stepsize * 1.5)).eval()
        # else:
        #     logger.log("kl just right!")

        logger.record_tabular("EpRewMean", np.mean([path["reward"].sum() for path in paths]))
        logger.record_tabular("EpRewSEM", np.std([path["reward"].sum()/np.sqrt(len(paths)) for path in paths]))
        logger.record_tabular("EpLenMean", np.mean([pathlength(path) for path in paths]))
        logger.record_tabular("KL", kl)
        print(do_std())
        if callback:
            callback()
        logger.dump_tabular()
        i += 1

    coord.request_stop()
    coord.join(enqueue_threads)

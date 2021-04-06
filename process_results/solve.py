import numpy as np
import mdptoolbox as mdp


def solve(env, discount=1.0, steps=np.PINF):
    nS = env.nS + 1
    nA = env.nA
    t = np.zeros((nA, nS, nS))
    r = np.zeros((nA, nS, nS))
    #has_terminals = False
    for a in range(nA):
        for s in range(nS - 1):
            line = env.P[s][a]
            for p_trans, s_p, rew, done in line:
                if done:
                    t[a, s, nS - 1] += 1.0
                    r[a, s, nS - 1] = rew
                    #has_terminals = True
                else:
                    t[a, s, s_p] += p_trans
                    r[a, s, s_p] = rew
            t[a, s, :] /= np.sum(t[a, s, :])
        t[a, nS - 1, nS - 1] = 1.0
    # if not has_terminals:
    #     t = t[:, :-1, :-1]
    #     r = r[:, :-1, :-1]
    if steps == np.PINF:
        m = mdp.mdp.PolicyIterationModified(transitions=t,
                                            reward=r, discount=discount, epsilon=0.00001, max_iter=1000)
        m.run()
        v = m.V
    else:
        m = mdp.mdp.FiniteHorizon(transitions=t, reward=r, discount=discount, N=steps)
        m.run()
        v = m.V[:, 0]
    result = np.sum([v[i] * env.isd[i] for i in range(nS - 1)])
    return result

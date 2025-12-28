from numba import njit


@njit(fastmath=True, cache=True, nogil=True)
def distributive_rhs(y, dy, A_i, B_i, C_i, D_i, E_i, tf_scale, TF_inputs, S_all,
                     offset_y, offset_s, n_sites):
    N = len(A_i)

    for i in range(N):
        y_start = offset_y[i]
        idx_R = y_start
        idx_P = y_start + 1

        s_start = offset_s[i]
        ns = n_sites[i]

        R = y[idx_R]
        P = y[idx_P]

        Ai = A_i[i]
        Bi = B_i[i]
        Ci = C_i[i]
        Di = D_i[i]
        Ei = E_i[i]

        # u = TF_inputs[i]
        # if u < -1.0: u = -1.0
        # if u > 1.0: u = 1.0
        # synth = Ai * (1.0 + tf_scale * u)

        u = TF_inputs[i]
        u = u / (1.0 + abs(u))  # maps to (-1,1)
        synth = Ai * (1.0 + tf_scale * u)

        dy[idx_R] = synth - Bi * R

        if ns == 0:
            dy[idx_P] = Ci * R - Di * P
        else:
            sum_S = 0.0
            sum_Ps = 0.0
            for j in range(ns):
                s_rate = S_all[s_start + j]
                ps_val = y[y_start + 2 + j]

                sum_S += s_rate
                sum_Ps += Ei * ps_val
                dy[y_start + 2 + j] = s_rate * P - (Ei + Di) * ps_val

            dy[idx_P] = Ci * R - (Di + sum_S) * P + sum_Ps

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import os

def _to_1d_float(seq):
        # Coerce each entry to a scalar float; handles 1x1 arrays, lists, etc.
        return np.array([float(np.asarray(v).reshape(-1)[0]) for v in seq], dtype=float)

def plot_Qt(Q_vals_BCUC, Q_vals_conformal, Q_vals_nll, Q_targets, dir_name, title="Q_t"):
    # --- style maps ---
    colors = {"BCUC": "tab:blue", "Conformal": "tab:orange", "NLL": "tab:green"}
    # cycle linestyles for persons (add more if you have more people)
    ls_list = ["-", "--", ":", "-.", (0,(5,2,1,2))]  # last is custom dash
    person_styles = ls_list if len(Q_vals_BCUC) <= len(ls_list) else ls_list + ["-"]*(len(Q_vals_BCUC)-len(ls_list))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlabel("Time Step")
    ax.set_title(title)
    ax.grid(True)

    for i, (Q_b, Q_c, Q_n) in enumerate(zip(Q_vals_BCUC, Q_vals_conformal, Q_vals_nll)):
        ls = person_styles[i]
        # coerce to 1D float arrays and align lengths defensively
        y_b  = _to_1d_float(Q_b)
        y_c  = _to_1d_float(Q_c)
        y_n  = _to_1d_float(Q_n)

        x_b = np.arange(len(y_b)); ax.plot(x_b, y_b, color=colors["BCUC"],     linestyle=ls, linewidth=1.8)
        x_c = np.arange(len(y_c)); ax.plot(x_c, y_c, color=colors["Conformal"], linestyle=ls, linewidth=1.8)
        x_n = np.arange(len(y_n)); ax.plot(x_n, y_n, color=colors["NLL"],       linestyle=ls, linewidth=1.8)

        # target line (thin gray so it doesn't dominate)
        q_star = float(np.asarray(Q_targets[i]).reshape(-1)[0])
        ax.axhline(y=q_star, color="0.5", linestyle="--", linewidth=1.0)

    # --- legends: (1) method-by-color, (2) person-by-linestyle ---
    method_handles = [plt.Line2D([0],[0], color=c, lw=2) for c in [colors["BCUC"], colors["Conformal"], colors["NLL"]]]
    method_labels  = ["BCUC (quantile-coupled)", "Conformal", "NLL (direct)"]
    leg1 = ax.legend(method_handles, method_labels, title="Method", loc="upper left")

    person_handles = [plt.Line2D([0],[0], color="black", lw=2, linestyle=ls) for ls in person_styles[:len(Q_vals_BCUC)]]
    person_labels  = [f"Person {p}" for p in Q_targets]  # or use your own labels
    leg2 = ax.legend(person_handles, person_labels, title="Person", loc="lower left")
    ax.add_artist(leg1)  # keep both legends

    # optional: place both legends outside if you prefer
    # leg1 = ax.legend(method_handles, method_labels, title="Method",
    #                  loc='center left', bbox_to_anchor=(1.02, 0.65))
    # ax.add_artist(leg1)
    # leg2 = ax.legend(person_handles, person_labels, title="Person",
    #                  loc='center left', bbox_to_anchor=(1.02, 0.25))

    fig.tight_layout()
    fig.savefig(os.path.join(dir_name, "Q_t_over_time_clean.png"), dpi=300, bbox_inches='tight')

    # usage (with your existing variables):
    # plot_Qt(Q_vals_BCUC, Q_vals_conformal, Q_vals_nll, persons, title="Q_t")

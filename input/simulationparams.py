import numpy as np
from dataclasses import dataclass, field


@dataclass
class SimuParams:
    training_time: float = 150  # training time, which could be in hours, days or weeks depending on scenario
    prediction_time: float = 50  # testing time
    mu_mean: float = 0.001 # mean baseline averaged over all nodes (and time if sinusoidal)
    mu_variation: float = 1.9  # applies a correlation between node features and node risk
    alpha: float = 0.12  # expected number of "direct" excitation events
    lifetime: float = 4  # average time to trigger event (1/beta)
    n_nodes: int = 2000  # number of nodes
    network_type: str = 'WS'  # type of graph, either BA or WS
    # number of features per node and proportion of each feature set to 1
    feature_proportions: list = field(default_factory=lambda: [0.1, 0.1, 0.1, 0.1, 0.1])
    homophilic: bool = False # neighbours have similar features if True
    row: float = 0
    omega: float = 1
    phi: float = 0
    run_time: float = field(init=False)

    def __post_init__(self):
        self.run_time = self.training_time + self.prediction_time


simu_params = {
    '5_contagious_correlated' : SimuParams(),
    '8_CC_scale_free' : SimuParams(alpha=0.08, mu_mean=0.0012, network_type='BA'),
    '6_contagious_homophilic' : SimuParams(homophilic=True),
    '7_CC_sinusoidal' : SimuParams(row=0.5, omega=0.05, phi=np.pi),
    '4_contagious' : SimuParams(mu_variation=0),
    '1_spontaneous' : SimuParams(alpha=0, mu_mean=0.002, mu_variation=0),
    '2_correlated' : SimuParams(alpha=0, mu_mean=0.002),
    '3_homophilic' : SimuParams(alpha=0, mu_mean=0.002, homophilic=True),
    '9_CC_high_frequency' : SimuParams(n_nodes=200, training_time=1500, prediction_time=500),
}

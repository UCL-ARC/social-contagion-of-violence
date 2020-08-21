import numpy as np
from dataclasses import dataclass, field


@dataclass
class Parameters:
    training_time: float = 150  # training time, which could be in hours, days or weeks depending on scenario
    prediction_time: float = 50  # testing time
    mean_mu: float = 0.001 # mean baseline averaged over all nodes (and time if sinusoidal)
    alpha: float = 0.15  # expected number of "direct" excitation events
    lifetime: float = 5  # average time between initial and triggered event (1/beta)
    n_nodes: int = 2000  # number of nodes
    network_type: str = 'newman_watts_strogatz'  # type of graph, either barabasi_albert or newman_watts_strogatz
    feature_proportions: list = field(default_factory=lambda: [0.1, 0.1, 0.1, 0.1, 0.1])
    feature_variation: float = 1.9  # applies a correlation between node features and node risk
    homophilic: bool = False
    row: float = 0
    omega: float = 1
    phi: float = 0
    run_time: float = field(init=False)

    def __post_init__(self):
        self.run_time = self.training_time + self.prediction_time


params = dict(
    contagious_correlated=Parameters(),
    scale_free=Parameters(alpha=0.08, mean_mu=0.0012, network_type='barabasi_albert'),
    contagious_homophilic=Parameters(homophilic=True),
    sinusoidal=Parameters(row=0.5, omega=0.05, phi=np.pi),
    contagious=Parameters(feature_variation=0),
    correlated=Parameters(alpha=0, mean_mu=0.002),
    homophilic=Parameters(alpha=0, mean_mu=0.002, homophilic=True),
    spontaneous=Parameters(alpha=0, mean_mu=0.002, feature_variation=0),
    high_frequency=Parameters(n_nodes=200, training_time=1500, prediction_time=500),
)

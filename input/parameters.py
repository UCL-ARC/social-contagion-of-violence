import numpy as np
from dataclasses import dataclass, field


@dataclass
class Parameters:
    training_time: float = 150  # training time, which could be in hours, days or weeks depending on scenario
    prediction_time: float = 50  # testing time
    average_events: float = 0.2  # expected number of random events averaged over training time and all nodes
    contagion: float = 0.1  # expected number of "direct" triggered events
    lifetime: float = 5  # average time between initial and triggered event
    n_nodes: int = 2000  # number of nodes
    network_type: str = 'barabasi_albert'  # type of graph, either barabasi_albert or newman_watts_strogatz
    feature_proportions: list = field(default_factory=lambda: [0.2, 0.2, 0.2, 0.2, 0.2])
    feature_variation: float = 2  # applies a correlation between node features and node risk
    homophilic: bool = False
    row: float = 0
    omega: float = 1
    phi: float = 0

    run_time = training_time + prediction_time


params = dict(
    contagion_correlation=Parameters(),
    newman_watts=Parameters(network_type='newman_watts_strogatz'),
    contagion_homophilic=Parameters(homophilic=True),
    sinusoidal=Parameters(row=0.5, omega=0.05, phi=np.pi),
    contagion_only=Parameters(feature_variation=0),
    correlation_only=Parameters(contagion=0, average_events=0.3),
    homophilic_only=Parameters(contagion=0, average_events=0.3, homophilic=True),
    homophilic_mimic=Parameters(feature_variation=0, contagion=0.5, lifetime=750),
    static=Parameters(contagion=0, feature_variation=0, average_events=0.3)
)

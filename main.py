from survival_metrics.metrics import get_ci_score, get_brier_score
import numpy as np

prediction = np.array([0.4, 0.3, 0.2, 0.1])
death = np.array([1, 1, 1, 1])
survival_time = np.array([1, 2, 3, 4])
time = np.array([10])
ci_score = get_ci_score(prediction, survival_time, death, time)

print(ci_score)
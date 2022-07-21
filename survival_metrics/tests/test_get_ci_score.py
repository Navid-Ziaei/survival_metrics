import unittest
import numpy as np
from survival_metrics.metrics import get_ci_score


class TestGetCIScore(unittest.TestCase):
    """
    unit test class for creating working folder function
    """

    def test_main(self):
        """
        the main test scenario
        :return: None
        """
        # test inputs
        # case 1 : all
        prediction = np.array([0.4, 0.3, 0.2, 0.1])
        death = np.array([1, 1, 1, 1])
        survival_time = np.array([1, 2, 3, 4])
        time = np.array([10])
        ci_score = get_ci_score(prediction, survival_time, death, time)
        self.assertAlmostEqual(ci_score, 1.0)

        # case 2 : all measurements before time horizon
        prediction = np.array([0.4, 0.3, 0.2, 0.1])
        death = np.array([1, 1, 1, 1])
        survival_time = np.array([4, 5, 6, 7])
        time = np.array([3])
        ci_score = get_ci_score(prediction, survival_time, death, time)
        self.assertAlmostEqual(ci_score, -1)

        # case 3 : all measurements censored
        prediction = np.array([0.4, 0.3, 0.2, 0.1])
        death = np.array([0, 0, 0, 0])
        survival_time = np.array([1, 2, 3, 4])
        time = np.array([3])
        ci_score = get_ci_score(prediction, survival_time, death, time)
        self.assertAlmostEqual(ci_score, -1)

        # case 4 :
        prediction = np.array([0.4, 0.5, 0.3, 0.2, 0.1])
        death = np.array([1, 1, 0, 1, 1])
        survival_time = np.array([1, 2, 3, 4, 5])
        time = np.array([13])
        ci_score = get_ci_score(prediction, survival_time, death, time)
        self.assertAlmostEqual(ci_score, 0.875)

        # case 5 :
        prediction = np.array([0.4, 0.2, 0.3, 0.1])
        death = np.array([0, 1, 1, 1])
        survival_time = np.array([1, 2, 3, 4])
        time = np.array([10])
        ci_score = get_ci_score(prediction, survival_time, death, time)
        self.assertAlmostEqual(ci_score, 0.666, places=2)

        # case 6 : Include Equal predictions
        prediction = np.array([0.4, 0.3, 0.2, 0.1, 0.1])
        death = np.array([1, 1, 1, 1, 1])
        survival_time = np.array([1, 2, 3, 4, 5])
        time = np.array([10])
        ci_score = get_ci_score(prediction, survival_time, death, time)
        self.assertAlmostEqual(ci_score, 0.95, places=3)

        # case 7 : All Equal predictions
        prediction = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        death = np.array([1, 1, 1, 1, 1])
        survival_time = np.array([1, 2, 3, 4, 5])
        time = np.array([10])
        ci_score = get_ci_score(prediction, survival_time, death, time)
        self.assertAlmostEqual(ci_score, 0.5, places=3)


if __name__ == '__main__':
    unittest.main(TestGetCIScore())
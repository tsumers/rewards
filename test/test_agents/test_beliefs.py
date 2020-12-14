import pandas as pd
import unittest

from science.agents.beliefs import IndependentNormals, PRIOR_MEAN, PRIOR_PRECISION


class MyTestCase(unittest.TestCase):

    def test_generate_gaussian_priors(self):

        prior = IndependentNormals(features=['pink'])
        self.assertDictEqual({"pink": {'mean': PRIOR_MEAN, "precision": PRIOR_PRECISION}}, prior.beliefs)

    def test_empty_belief_udpate(self):

        prior = IndependentNormals(features=['pink'])
        empty_df = pd.DataFrame()
        posterior = prior.literal_update(empty_df)

        self.assertDictEqual(prior.beliefs, posterior.beliefs)

    def test_literal_update_precision(self):

        prior = IndependentNormals(features=['pink', 'circle'], precision=PRIOR_PRECISION)

        obs_precision = 1
        obs = pd.DataFrame.from_records([{"feature": "pink", "mean": 0, "precision": obs_precision}])

        posterior = prior.literal_update(obs)

        self.assertEqual(posterior["pink"]["precision"], PRIOR_PRECISION + obs_precision)
        self.assertEqual(posterior["circle"]["precision"], PRIOR_PRECISION)

    def test_literal_update_mean(self):

        prior = IndependentNormals(features=['pink', 'circle'], precision=PRIOR_PRECISION)

        obs_mean = 2
        obs_precision = PRIOR_PRECISION
        obs = pd.DataFrame.from_records([{"feature": "pink",
                                          "mean": obs_mean,
                                          "precision": obs_precision}])

        posterior = prior.literal_update(obs)

        self.assertEqual(posterior["pink"]["mean"], (PRIOR_MEAN + obs_mean) / 2)
        self.assertEqual(posterior["circle"]["mean"], PRIOR_MEAN)

    def test_pragmatic_update(self):

        prior = IndependentNormals(features=['pink', 'circle'], precision=PRIOR_PRECISION)

        obs_mean = 2
        obs_precision = PRIOR_PRECISION
        obs = pd.DataFrame.from_records([{"feature": "pink",
                                          "mean": obs_mean,
                                          "precision": obs_precision}])

        literal_posterior = prior.literal_update(obs)

        pragmatic_precision = 2
        pragmatic_posterior = prior.pseudopragmatic_gaussian_update(obs, pragmatic_precision)

        # Literal and pragmatic should be the same for means of both, and variance of pink
        self.assertEqual(literal_posterior["pink"]["mean"], pragmatic_posterior["pink"]["mean"])
        self.assertEqual(literal_posterior["pink"]["precision"], pragmatic_posterior["pink"]["precision"])
        self.assertEqual(literal_posterior["circle"]["mean"], pragmatic_posterior["circle"]["mean"])

        # Should *differ* for precision of unmentioned (circle) feature
        self.assertNotEqual(literal_posterior["circle"]["precision"], pragmatic_posterior["circle"]["precision"])
        self.assertEqual(pragmatic_posterior["circle"]["precision"], PRIOR_PRECISION + pragmatic_precision)

    def test_multiple_literal_updates(self):

        prior = IndependentNormals(features=['pink', 'circle'], precision=PRIOR_PRECISION)

        obs_mean = 2
        obs_precision = PRIOR_PRECISION
        obs = pd.DataFrame.from_records([{"feature": "pink",
                                          "mean": obs_mean,
                                          "precision": obs_precision},
                                         {"feature": "circle",
                                          "mean": obs_mean,
                                          "precision": obs_precision},
                                         ])

        posterior = prior.literal_update(obs)

        self.assertAlmostEqual(posterior["pink"]["mean"], (PRIOR_MEAN + obs_mean) / 2)
        self.assertAlmostEqual(posterior["circle"]["mean"], (PRIOR_MEAN + obs_mean) / 2)
        self.assertAlmostEqual(posterior["circle"]["precision"], PRIOR_PRECISION + obs_precision)

        posterior = posterior.literal_update(obs)

        self.assertAlmostEqual(posterior["pink"]["mean"], (PRIOR_MEAN + 2 * obs_mean) / 3)
        self.assertAlmostEqual(posterior["circle"]["mean"], (PRIOR_MEAN + 2 * obs_mean) / 3)
        self.assertAlmostEqual(posterior["circle"]["precision"], PRIOR_PRECISION + 2 * obs_precision)

    def test_belief_samples(self):

        mean = 0
        var = 2

        prior = IndependentNormals(features=['pink'], mean=0, precision=1 / var)
        samples = prior.sample_beliefs(n=100000)

        self.assertAlmostEqual(mean, abs(samples['pink'].mean()), 1)
        self.assertAlmostEqual(var, samples['pink'].var(), 1)


if __name__ == '__main__':
    unittest.main()

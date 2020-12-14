import unittest

from science.agents.agents import MultivariateNormalLearner, StatefulFeedforwardNeuralAgent

from science.analysis.model_benchmarking import offline_benchmark_levels, run_experiment_dyads, \
    run_models_on_human_human_data, human_human_experiment_dataframe

EXPERIMENT_DF = human_human_experiment_dataframe()

# Test all (qualitatively different) models to ensure stability
MODELS_TO_TEST = [MultivariateNormalLearner(), StatefulFeedforwardNeuralAgent()]


class MyTestCase(unittest.TestCase):

    def test_shape_magnitude_test_levels(self):

        config = {"shape_config": "^_s_o", "color_config": "y_c_m"}
        levels = offline_benchmark_levels(config)
        two_pts = levels[0].object_df[levels[0].object_df.reward == 2].iloc[0]

        # Should be a pink triangle under this config
        self.assertEqual(1, two_pts["pink|triangle"])
        self.assertEqual(0, two_pts["yellow|square"])

        config = {"shape_config": "s_o_^", "color_config": "m_c_y"}
        levels = offline_benchmark_levels(config)
        two_pts = levels[0].object_df[levels[0].object_df.reward == 2].iloc[0]

        # Should be a yellow square under this config
        self.assertEqual(0, two_pts["pink|triangle"])
        self.assertEqual(1, two_pts["yellow|square"])

    def test_experiment_dyads(self):

        test_config = EXPERIMENT_DF.value_mask_config.sample(1).iloc[0]
        test_dyad = EXPERIMENT_DF.task_uuid.sample(1).iloc[0]
        test_dyad_sequence = EXPERIMENT_DF[EXPERIMENT_DF.task_uuid == test_dyad]

        # Just ensure this runs without an exception
        run_experiment_dyads(MODELS_TO_TEST, test_dyad_sequence, value_mask_config=test_config)

    def test_run_models_on_exp_data(self):

        test_config = EXPERIMENT_DF.value_mask_config.sample(1).iloc[0]

        # Run an abbreviated test: just 3 steps, and just 10 benchmark levels per step
        test_games = EXPERIMENT_DF.sample(3)
        benchmark_levels = offline_benchmark_levels(test_config)[:10]

        run_models_on_human_human_data(MODELS_TO_TEST, test_games, benchmark_levels, test_config)


if __name__ == '__main__':
    unittest.main()





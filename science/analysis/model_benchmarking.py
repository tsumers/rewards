import copy
import json
import pandas as pd

from pathlib import Path

from science.environments.environments import Environment
from science.agents.actions import Trajectory
from science.analysis.data_management import switch_chat_features, get_all_reward_configs

test_levels_path = Path(__file__).parent / "../../notebooks/data/exp_test_levels.json"


def pilot_benchmark_levels():

    with test_levels_path.open('r') as f:
        test_levels_json = json.loads(f.read())

    return [Environment.from_pilot_config(t) for t in test_levels_json]


def all_offline_benchmark_permutations():
    """Return all possible versions of 100 generated levels in a dictionary keyed by reward function."""

    reward_configs = get_all_reward_configs()
    benchmark_dictionary = {}
    for r in reward_configs:
        key = reward_configs[r]["color_config"] + "|" + reward_configs[r]["shape_config"]
        benchmark_dictionary[key] = offline_benchmark_levels(reward_configs[r])

    return benchmark_dictionary


def offline_benchmark_levels(value_mask_config):
    """Return a list of 100 generated levels to benchmark an agent against, configured with this reward function."""

    with test_levels_path.open('r') as f:
        test_levels_json = json.loads(f.read())

    return [Environment.from_shape_magnitude_config(t, value_mask_config) for t in test_levels_json]


def human_human_experiment_dataframe():
    """Load results from original human-human gameplay."""

    experiment_data_path = Path(__file__).parent / "../../notebooks/data/human_trial_data.json"
    with experiment_data_path.open('r') as f:
        exp_json = json.loads(f.read())

        exp_df = pd.DataFrame.from_records(exp_json)
        exp_df = exp_df[exp_df.comm_viz_cond == "chat|full"]
        exp_df.chat_text = exp_df.chat_text.apply(lambda x: x if x is not None else "")

    exp_df["value_mask_config"] = exp_df["game"].apply(lambda x: x["value_mask_config"])
    return exp_df[["task_uuid", "level_number", "chat_text", "game", "pct_max_score", "player_score",
                       "cum_player_score", "cum_max_score", "value_mask_config", "comm_viz_cond"]]


def pilot_dataframe():

    pilot_data_path = Path(__file__).parent / "../../notebooks/data/cogsci_experiment_games.json"
    with pilot_data_path.open('r') as f:
        pilot_exp_json = json.loads(f.read())

    pilot_exp_df = pd.DataFrame.from_records(pilot_exp_json)
    relevant_condition_df = pilot_exp_df[pilot_exp_df.comm_viz_cond == "chat|full"]

    return relevant_condition_df[["task_uuid", "level_number", "chat_text", "game", "pct_max_score",
                                  "player_score", "cum_player_score", "cum_max_score", "color_mask"]]


def run_models_on_human_human_data(models, human_human_exp_df, benchmark_levels, value_mask_config):
    """Given a list of models and a slice of CogSci dataframe with records, run eval against them."""

    results = []

    for m in models:
        m.reset_beliefs()
        results.extend(eval_model_against_test_levels(m, benchmark_levels, {"iteration": 0})
)
    for i, (_, row) in enumerate(human_human_exp_df.iterrows()):

        # Generate appropriately-reconfigured trajectory and feedback
        switched_utterance = switch_chat_features(row["chat_text"],
                                         row["value_mask_config"],
                                         value_mask_config)

        trajectory = Trajectory.from_exp_psiturk_record(row["game"], value_mask_config=value_mask_config)

        iteration_info = {"iteration": i+1, "level_number": row["level_number"], "task_id": row["task_uuid"]}

        for m in models:
            m.update_beliefs(switched_utterance, trajectory)
            model_results = eval_model_against_test_levels(m, benchmark_levels, iteration_info)
            results.extend(model_results)

    return pd.DataFrame.from_records(results)


def run_models_on_pilot_data(models, pilot_exp_df, benchmark_levels, verbose=False):
    """Given a list of models and a slice of a pilot experiment dataframe, run eval against them."""

    results = []

    for m in models:
        m.reset_beliefs()
        results.extend(eval_model_against_test_levels(m, benchmark_levels, {"iteration": 0}))

    results = []

    for i, (_, row) in enumerate(pilot_exp_df.iterrows()):

        if verbose:
            print("Learning from round {} --> \"{}\"".format(i + 1, row["chat_text"]))

        utterance = row["chat_text"]
        trajectory = Trajectory.from_pilot_exp_psiturk_record(row["game"])
        iteration_info = {"iteration": i + 1, "level_number": row["level_number"], "task_id": row["task_uuid"]}

        for m in models:
            m.update_beliefs(utterance, trajectory)
            model_results = eval_model_against_test_levels(m, benchmark_levels, iteration_info)
            results.extend(model_results)

    return pd.DataFrame.from_records(results)


def run_experiment_dyads(models, experiment_df, value_mask_config=None, n=10):

    full_results = []
    for k, g in experiment_df.groupby("task_uuid"):
        print("Running dyad {} ({} levels)".format(k, len(g)))
        dyad_results = run_dyad(models, g, value_mask_config=value_mask_config, n=n)
        full_results.extend(dyad_results)

    return pd.DataFrame.from_records(full_results)


def run_dyad(models, dyad_df, value_mask_config=None, n=10):

    for m in models:
        m.reset_beliefs()

    results = []

    dyad_df = dyad_df.sort_values("level_number", ascending=True)
    for i, (_, row) in enumerate(dyad_df.iterrows()):

        true_trajectory = Trajectory.from_exp_psiturk_record(row["game"], value_mask_config=value_mask_config)

        iteration_info = {"iteration": i, "level_number": row["level_number"], "task_uuid": row["task_uuid"]}
        human_results = dict(copy.deepcopy(iteration_info), **{"model": "human",
                                                               "reward": true_trajectory.reward,
                                                               "pct_max": true_trajectory.pct_max_reward})
        results.append(human_results)

        for m in models:
            # First, run the model on the level to get its score
            t = m.execute_trajectories(true_trajectory.environment, n=n)
            model_results = dict(copy.deepcopy(iteration_info),
                                 **{"model": "{}".format(m), "reward": sum(t[0])/n, "pct_max": sum(t[1])/n})
            results.append(model_results)

            switched_utterance = switch_chat_features(row["chat_text"],
                                                      row["value_mask_config"],
                                                      value_mask_config)

            # Then learn from the actual chat text
            m.update_beliefs(switched_utterance, true_trajectory)

    return results


def eval_model_against_test_levels(model, benchmark_levels, base_dict):

    test_trajectories = (model.execute_trajectory(level) for level in benchmark_levels)

    # Return a list of dictionaries with the model name + learning trajectory
    return [dict(copy.deepcopy(base_dict),
                 **{"model": "{}".format(model), "reward": t.reward, "pct_max": t.pct_max_reward})
            for t in test_trajectories]






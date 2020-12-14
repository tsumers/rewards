import copy
import pandas as pd
import numpy as np

import matplotlib.pylab as plt
from matplotlib.colors import LinearSegmentedColormap

import seaborn as sns

from science.utils import FEATURE_ABBREVIATION_DICT, CONJUNCTIONS

def feature_magnitude_dict(three_feature_list):
    """feature list should be in order small, medium, large"""

    values_list = [[1, 2, 3],
                    [4, 5, 6, 7],
                    [7, 8, 9, 10]]

    feature_dict = {}
    for feature, values in zip(three_feature_list, values_list):

        for value in values:
            feature_dict[value] = feature
            feature_dict[-value] = feature

    return feature_dict


def feature_sign_dict(three_feature_list):
    """feature list should be in order -/0/+"""

    feature_dict = {}

    for i in list(range(1, 11)):
        feature_dict[-i] = three_feature_list[0]

    feature_dict[0] = three_feature_list[1]

    for i in list(range(1, 11)):
        feature_dict[i] = three_feature_list[2]

    return feature_dict


def plot_experiment_colors(original_object_list, color_dict, shape_dict):

    object_list = copy.deepcopy(original_object_list)

    for o in object_list:
        o["true_value"] = o.get('value')
        if o.get('color_value') is None:
            o["color_value"] = o.get('true_value')

    objects_df = pd.DataFrame(object_list)

    objects_df["color"] = objects_df.apply(lambda x: color_dict[x["true_value"]], axis=1)
    objects_df["shape"] = objects_df.apply(lambda x: shape_dict[x["color_value"]] if x["color_value"] > -99 else shape_dict[x["true_value"]], axis=1)

    for s, df in objects_df.groupby("shape"):
        plt.scatter(df.x, df.y, s=600, marker=s, c="k")
        plt.scatter(df.x, df.y, s=400,
                    marker=s,
                    color=df.color.values)


def plot_gameplay_dataframe(objects_df, first_trajectory, second_trajectory=None, title="", value_mask_config=None,
                            alpha=1, first_trajectory_color="copper", second_trajectory_color="bone"):

    cluster_str = "Clusters (clockwise): "
    for k, g in objects_df.groupby("cluster_id", sort=True):
        cluster_str += "{}, ".format(g[g.value >= 0].value.sum())

    plt.figure(figsize=(10.0, 10.0))
    plt.gca().set_facecolor('k')

    # Plot player trajectory
    plt.scatter(first_trajectory.x, first_trajectory.y, c=first_trajectory.ts, cmap=first_trajectory_color, s=300,
                alpha=alpha)

    # Plot teacher trajectory
    if second_trajectory is not None and not second_trajectory.empty:
        plt.scatter(second_trajectory.x, second_trajectory.y, c=second_trajectory.ts, cmap=second_trajectory_color,
                    s=300, alpha=alpha)

    if value_mask_config is not None:
        color_dict = feature_sign_dict(value_mask_config["color_config"].split("_"))
        shape_dict = feature_magnitude_dict(value_mask_config["shape_config"].split("_"))
        plot_experiment_colors(objects_df.to_dict(orient="records"), color_dict, shape_dict)
    else:
        plot_object_layout(objects_df.to_dict(orient="records"), ["r", 'w', 'g'], label_values=True,
                           zero_shape="s", positive_shape='s', negative_shape='s',  plot_fake_colors=False)
        # plot_object_layout(objects_df.to_dict(orient="records"), ["y", 'w', 'c', 'w', "m"], zero_shape="^")

    plt.xlim(0, 75)
    plt.ylim(75, 0)

    plt.suptitle(title + "\n" + cluster_str)

    plt.tick_params(
        axis='both',
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,
        left=False,
        labelleft=False
    )

    plt.show()


def plot_object_layout(original_object_list, object_colormap, positive_shape='o', negative_shape='s', zero_shape=None,
                       color_magnitude=False, label_values=False, max_obj_value=10, plot_fake_colors=True):
    object_list = copy.deepcopy(original_object_list)

    for o in object_list:
        o["true_value"] = o.get('value')

        if not plot_fake_colors:
            o["color_value"] = o.get('true_value')
        if o.get('color_value') is None or np.isnan(o.get('color_value')):
            o["color_value"] = o.get('true_value')

    objects_df = pd.DataFrame(object_list)

    if color_magnitude:
        objects_df["color_value"] = abs(objects_df.color_value)
        min_obj_value = 0

    else:
        objects_df["color_value"] = objects_df.color_value
        min_obj_value = -max_obj_value

    colormap_for_plot = LinearSegmentedColormap.from_list("val_cmap", object_colormap)

    pos_objects = objects_df[objects_df.true_value > 0]
    plt.scatter(pos_objects.x, pos_objects.y, s=600, marker=positive_shape, c="k")
    plt.scatter(pos_objects.x, pos_objects.y, s=400, marker=positive_shape,
                c=pos_objects.color_value, cmap=colormap_for_plot, vmin=min_obj_value, vmax=max_obj_value)

    neg_objects = objects_df[objects_df.true_value < 0]
    plt.scatter(neg_objects.x, neg_objects.y, s=600, marker=negative_shape, c="k")
    plt.scatter(neg_objects.x, neg_objects.y, s=400, marker=negative_shape,
                c=neg_objects.color_value, cmap=colormap_for_plot, vmin=min_obj_value, vmax=max_obj_value)

    zero_objects = objects_df[objects_df.true_value == 0]

    if zero_shape is None:
        # Default: plot zero-valued objects as positive shape, regular color
        plt.scatter(neg_objects.x, neg_objects.y, s=600, marker=positive_shape, c="k")
        plt.scatter(zero_objects.x, zero_objects.y, s=400, marker=positive_shape,
                    c=zero_objects.color_value, cmap=colormap_for_plot, vmin=min_obj_value, vmax=max_obj_value)

    else:
        # If "zero shape" is passed in, make it a distractor class
        plt.scatter(zero_objects.x, zero_objects.y, s=600, marker=zero_shape, c="k")
        plt.scatter(zero_objects.x, zero_objects.y, s=400, marker=zero_shape,
                    c=zero_objects.color_value, cmap=colormap_for_plot, vmin=min_obj_value, vmax=max_obj_value)

    if label_values:
        for index, row in objects_df.iterrows():
            plt.text(row["x"] - 1, row["y"] + .5, int(row["true_value"]), weight='bold')


def plot_pilot_game_record(g, teacher_feedback=None, teacher_demo=None):
    locations = [{"ts": e[0], "x": e[1], "y": e[2]} for e in g["player_locations"]]
    player_df = pd.DataFrame(locations)

    objects_df = pd.DataFrame(g["config"]["frame_data"]["objects"])

    game_events = []
    for event in g["game_events"]:
        game_events.append({
            "ts": event["ms_elapsed"],
            "x": event["data"]["object"]["x"],
            "y": event["data"]["object"]["y"],
            "value": event["data"]["object"]["value"],
            "score": event["data"]["score"]
        })
    events_df = pd.DataFrame(game_events)

    teaching_act = ""
    if teacher_feedback is not None:
        teaching_act = "Teacher Feedback: {}".format(teacher_feedback)

    # This is stupid, but it winds up being np.nan which is annoyingly hard to check for.
    # So let's be pythonic and just try / except it...

    if teacher_demo is not None:
        teaching_act = "Teacher Score: {}".format(teacher_demo.get("score"))
        teacher_demo = pd.DataFrame([{"ts": e[0], "x": e[1], "y": e[2]} for e in teacher_demo["player_locations"]])

    title = "Level {}\nPlayer Score {}\n{}".format(g["level"], g["score"], teaching_act)
    plot_gameplay_dataframe(objects_df, player_df, second_trajectory=teacher_demo, title=title)


# Plot a series of games occuring on different levels.
# This takes a bunch of rows from the primary gameplay dataframe ("gp_df") and plots each
# individual player separately.
def plot_and_analyze(df_rows, print_analysis=False):
    for index, row in df_rows.iterrows():

        if row.get("chat_text") is not np.nan:
            chat_text = row.get("chat_text")
        else:
            chat_text = None

        if row.get("game_teacher") is not np.nan:
            teacher_game = row.get("game_teacher")
        else:
            teacher_game = None

        plot_pilot_game_record(row.game, chat_text, teacher_game)
        # cluster_analysis(row.game, print_analysis=print_analysis)


# Plot a series of traces on the same level.
# This can take two separate series of games ("gp_df.game" objects) and overlays all of the traces
# on the same level. You can control the colors of each series separately to contrast conditions, etc.
def overlay_traces(games_on_level, teaching_games_on_level=None, title="", **kwargs):
    locations = []
    for g in games_on_level:
        locations.extend([{"ts": e[0], "x": e[1], "y": e[2]} for e in g["player_locations"]])
    locations_df = pd.DataFrame(locations)

    teacher_locations = []
    if teaching_games_on_level is not None:
        for g in teaching_games_on_level:
            teacher_locations.extend([{"ts": e[0], "x": e[1], "y": e[2]} for e in g["player_locations"]])
    teacher_locations_df = pd.DataFrame(teacher_locations)

    objects_df = pd.DataFrame(games_on_level.iloc[0]["config"]["frame_data"]["objects"])

    plot_gameplay_dataframe(objects_df, locations_df, second_trajectory=teacher_locations_df,
                            title=title, **kwargs)


def plot_experiment_df_row(row):

    overlay_traces(row["game"], alpha=1, first_trajectory_color='bone',
                   value_mask_config=row["value_mask_config"].iloc[0])


def plot_learner_beliefs(learner_agent, title=None):

    plt.figure(figsize=(8, 5))
    sns.violinplot(data=learner_agent.belief_state.sample_beliefs())

    if title is None:
        title = learner_agent

    plt.suptitle(title)
    plt.ylim(-10, 10)
    plt.xlabel("Learner Belief over Feature Values")
    plt.ylabel("Feature Reward Estimate")
    plt.xticks(rotation=45)


def plot_trajectory(trajectory):

    ax = _get_new_canvas()

    plot_actions(trajectory, ax=ax)
    plot_environment(trajectory.environment, ax=ax)


def plot_actions(trajectory, ax):

    action_line = trajectory.to_points()

    ax.scatter(action_line["x"], action_line["y"], s=1000, marker='s', c='w', alpha=.4)
    ax.plot(action_line["x"], action_line["y"], c='w')


def _get_new_canvas():

    plt.figure(figsize=(10.0, 10.0))
    plt.gca().set_facecolor('k')
    plt.xlim(0, 75)
    plt.ylim(75, 0)

    plt.tick_params(
        axis='both',
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,
        left=False,
        labelleft=False
    )

    return plt.gca()


def plot_environment(environment, ax=None):

    if ax is None:
        ax = _get_new_canvas()

    inv_map = {v: k for k, v in FEATURE_ABBREVIATION_DICT.items()}

    for conjunction in CONJUNCTIONS:

        corresponding_objects = environment.object_df[environment.object_df[conjunction] == 1]
        for i, object_row in corresponding_objects.iterrows():

            color = inv_map[conjunction.split('|')[0]]
            shape = inv_map[conjunction.split('|')[1]]

            ax.scatter(object_row.x, object_row.y, s=600, marker=shape, c="k")
            ax.scatter(object_row.x, object_row.y, s=400, marker=shape, c=color)

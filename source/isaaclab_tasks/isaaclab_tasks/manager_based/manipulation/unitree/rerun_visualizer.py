import os
import json
import cv2
import time
import rerun as rr
import rerun.blueprint as rrb
from datetime import datetime
os.environ["RUST_LOG"] = "error"

import numpy as np
import rerun as rr

class RerunLogger:
    def __init__(self, rewards_plots=[], prefix = "", IdxRangeBoundary = 3000, memory_limit = "512MB"):
        self.prefix = prefix
        self.plots = rewards_plots
        self.IdxRangeBoundary = IdxRangeBoundary
        rr.init(datetime.now().strftime("Runtime_%Y%m%d_%H%M%S"))
        if memory_limit:
            rr.spawn(memory_limit = memory_limit, hide_welcome_screen = True)
        else:
            rr.spawn(hide_welcome_screen = True)

        # Set up blueprint for live visualization
        if self.IdxRangeBoundary:
            self.setup_blueprint()

    def setup_blueprint(self):
        views = []

        view = rrb.TimeSeriesView(
            origin = "/rewards",
            time_ranges=[
                rrb.VisibleTimeRange(
                    "step",
                    start = rrb.TimeRangeBoundary.cursor_relative(seq = -self.IdxRangeBoundary),
                    end = rrb.TimeRangeBoundary.cursor_relative(),
                )
            ],
            plot_legend = rrb.PlotLegend(visible = True),
            axis_y=rrb.ScalarAxis(range=(0, 1))
        )
        views.append(view)

        view = rrb.TimeSeriesView(
            origin = "/right_arm",
            time_ranges=[
                rrb.VisibleTimeRange(
                    "step",
                    start = rrb.TimeRangeBoundary.cursor_relative(seq = -self.IdxRangeBoundary),
                    end = rrb.TimeRangeBoundary.cursor_relative(),
                )
            ],
            plot_legend = rrb.PlotLegend(visible = True),
            axis_y=rrb.ScalarAxis(range=(-1, 1))
        )
        views.append(view)

        view = rrb.TimeSeriesView(
            origin = "/right_hand",
            time_ranges=[
                rrb.VisibleTimeRange(
                    "step",
                    start = rrb.TimeRangeBoundary.cursor_relative(seq = -self.IdxRangeBoundary),
                    end = rrb.TimeRangeBoundary.cursor_relative(),
                )
            ],
            plot_legend = rrb.PlotLegend(visible = True),
            axis_y=rrb.ScalarAxis(range=(-1, 1))
        )
        views.append(view)

        view = rrb.TimeSeriesView(
            origin = "/rewards_env0",
            time_ranges=[
                rrb.VisibleTimeRange(
                    "step",
                    start = rrb.TimeRangeBoundary.cursor_relative(seq = -self.IdxRangeBoundary),
                    end = rrb.TimeRangeBoundary.cursor_relative(),
                )
            ],
            plot_legend = rrb.PlotLegend(visible = True),
            axis_y=rrb.ScalarAxis(range=(0, 20))
        )
        views.append(view)

        view = rrb.Spatial2DView(
            origin = "/image_env0",
            time_ranges=[
                    rrb.VisibleTimeRange(
                        "step",
                        start = rrb.TimeRangeBoundary.cursor_relative(seq = -self.IdxRangeBoundary),
                        end = rrb.TimeRangeBoundary.cursor_relative(),
                    )
                ],
            )
        views.append(view)

        view = rrb.Spatial2DView(
            origin = "/image2_env0",
            time_ranges=[
                    rrb.VisibleTimeRange(
                        "step",
                        start = rrb.TimeRangeBoundary.cursor_relative(seq = -self.IdxRangeBoundary),
                        end = rrb.TimeRangeBoundary.cursor_relative(),
                    )
                ],
            )
        views.append(view)

        # image_plot_paths = [
        #                     f"{self.prefix}colors/color_0",
        #                     f"{self.prefix}colors/color_1",
        #                     f"{self.prefix}colors/color_2",
        #                     f"{self.prefix}colors/color_3"
        # ]
        # for plot_path in image_plot_paths:
        #     view = rrb.Spatial2DView(
        #         origin = plot_path,
        #         time_ranges=[
        #             rrb.VisibleTimeRange(
        #                 "idx",
        #                 start = rrb.TimeRangeBoundary.cursor_relative(seq = -self.IdxRangeBoundary),
        #                 end = rrb.TimeRangeBoundary.cursor_relative(),
        #             )
        #         ],
        #     )
        #     views.append(view)

        grid = rrb.Grid(contents = views,
                        grid_columns=2,               
                        column_shares=[1, 1],
                        row_shares=[1, 1], 
        )
        views.append(rr.blueprint.SelectionPanel(state=rrb.PanelState.Collapsed))
        views.append(rr.blueprint.TimePanel(state=rrb.PanelState.Collapsed))
        rr.send_blueprint(grid)


    def log_data(self, data: dict):
        rr.set_time("step", sequence=data.get('step', 0))

        rewards = data.get('rewards', {}) or {}
        for reward_key, reward_val in rewards.items():
            rr.log(f"/rewards/{reward_key}", rr.Scalars(reward_val), rr.SeriesLines(widths=[2]))

        # # Log rewards per environment 0
        rewards = data.get('rewards_env0', {}) or {}
        for reward_key, reward_val in rewards.items():
            rr.log(f"/rewards_env0/{reward_key}", rr.Scalars(reward_val), rr.SeriesLines(widths=[2]))


    def log_right_arm(self, name: str, joint_dict: dict):
        for joint_name, joint_val in joint_dict.items():
            rr.log(f"/right_arm/{name}/{joint_name}", rr.Scalars(joint_val), rr.SeriesLines(widths=[2]))

    def log_right_hand(self, name: str, joint_dict: dict):
        for joint_name, joint_val in joint_dict.items():
            rr.log(f"/right_hand/{name}/{joint_name}", rr.Scalars(joint_val), rr.SeriesLines(widths=[2]))

    def log_image(self, name: str, image):
        rr.log("/image_env0/" + name, rr.Image(image))

    def log_image2(self, name: str, image):
        rr.log("/image2_env0/" + name, rr.Image(image))


        # Log states
        # states = item_data.get('states', {}) or {}
        # for part, state_info in states.items():
        #     if part != "body" and state_info:
        #         values = state_info.get('qpos', [])
        #         for idx, val in enumerate(values):
        #             rr.log(f"{self.prefix}{part}/states/qpos/{idx}", rr.Scalar(val))

        # # Log actions
        # actions = item_data.get('actions', {}) or {}
        # for part, action_info in actions.items():
        #     if part != "body" and action_info:
        #         values = action_info.get('qpos', [])
        #         for idx, val in enumerate(values):
        #             rr.log(f"{self.prefix}{part}/actions/qpos/{idx}", rr.Scalar(val))

        # # Log colors (images)
        # colors = item_data.get('colors', {}) or {}
        # for color_key, color_val in colors.items():
        #     if color_val is not None:
        #         rr.log(f"{self.prefix}colors/{color_key}", rr.Image(color_val))



if __name__ == "__main__":
    visualizer = RerunLogger(
        rewards_plots=["reaching_object", "lifting_object", "move_to_target"],
    )

    # Simulate logging data
    for step in range(100):
        sample_data = {
            'step': step,
            'rewards': {
                'reaching_object': step * 0.1,
                'lifting_object': step * 0.05,
                'move_to_target': step * 0.2,
            }
        }
        visualizer.log_data(sample_data)
        time.sleep(0.1)  # Simulate time delay between data logs
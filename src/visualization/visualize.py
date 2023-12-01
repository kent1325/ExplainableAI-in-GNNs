import matplotlib.pyplot as plt
import numpy as np
import datetime
import os
from settings.config import ROOT_PATH
import visualization.plot_settings


class Plot:
    """Parent class for plotting.

    Encapsulates all the necessary utilities needed in all plot subclasses.
    """

    def __init__(self, x_label, y_label, title):
        self.x_label = x_label
        self.y_label = y_label
        self.title = title

    def plot(self) -> object:
        """Creates and returns a common canvas for all plotting types.

        Returns:
            fig (Object): figure object from matplotlib plot library.
            ax (object): figure object from matplotlib plot library.
        """
        # Create canvas
        fig, ax = plt.subplots()

        # Title and axis customization
        ax.set_title(self.title)
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)

        # Customization
        ax.grid(True)
        ax.tick_params(left=False, bottom=False, color="black")

        return fig, ax

    def export_figure(self, filename: str, overwrite: bool = True) -> None:
        """Exports the plot into a chosen folder.

        Args:
            filename (str): The name of the figure.
            overwrite (bool): Chooses saving behaviour -> overwrite existing (if any) plots or always create a new one
        """
        date = datetime.date.today().strftime("%Y%m%d")
        dest_folder = f"{ROOT_PATH}/reports/figures/{date}/"

        if "/" in filename:
            splitted_filename = filename.split("/")
            path = "/".join(splitted_filename[:-1])
            filename = splitted_filename[-1]
            dest_folder += path

        # Create folder if it does not exist
        os.makedirs(dest_folder, exist_ok=True)

        # Create full path
        version = 1
        file_extension = ".png"
        full_path = os.path.join(
            dest_folder, f"{filename}_ver_{version}{file_extension}"
        )

        # Increment version number if version already exists
        while not overwrite and os.path.exists(full_path):
            version += 1
            full_path = os.path.join(
                dest_folder, f"{filename}_ver_{version}{file_extension}"
            )

        # Save the figure
        plt.savefig(full_path, bbox_inches="tight")
        print(f"Successfully exported '{filename}_ver_{version}{file_extension}'")


class LinePlot(Plot):
    """Subclass for plotting lineplot

    Args:
        Plot (Class): Inherits methods and attributes from Plot
    """

    def __init__(self, x_label: str, y_label: str, title: str) -> None:
        super().__init__(x_label, y_label, title)

    def single_lineplot(self, x, label) -> None:
        # Create canvas
        fig, ax = super().plot()

        # Input values
        ax.plot(x, marker="o", markersize=3, linewidth=1.5, label=label)

        # Set legend
        fig.legend(
            loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes
        )

    def multi_lineplot(self, lines, labels) -> None:
        # Create canvas
        fig, ax = super().plot()

        # Input values
        for i in range(len(lines)):
            ax.plot(lines[i], marker="o", markersize=3, linewidth=1.5, label=labels[i])

        # Set legend
        fig.legend(
            loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes
        )


class CAMPlot(Plot):
    """Subclass for plotting CAM graphs

    Args:
        Plot (Class): Inherits methods and attributes from Plot
    """

    def __init__(self, x_label: str, y_label: str, title: str) -> None:
        super().__init__(x_label, y_label, title)

    def single_graph(self, exp, y_pred, y_true, y_original_pred) -> None:
        # Create canvas
        fig, ax = super().plot()

        # Input values
        exp.visualize_graph(ax=ax)
        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()
        ax.text(xmin, ymax - 0.1 * (ymax - ymin), f"Label = {y_true}")
        if y_original_pred is not None:
            ax.text(
                xmin,
                ymax - 0.2 * (ymax - ymin),
                f"Original Prediction  = {y_original_pred}",
            )
            ax.text(xmin, ymax - 0.15 * (ymax - ymin), f"Masked Prediction  = {y_pred}")
        else:
            ax.text(xmin, ymax - 0.15 * (ymax - ymin), f"Prediction  = {y_pred}")

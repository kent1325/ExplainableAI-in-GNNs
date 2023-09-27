import matplotlib.pyplot as plt
import numpy as np
import datetime
import os

#from settings.config import ROOT_PATH
#import visualization.plot_settings

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
        
    def _export_figure(self, filename: str, overwrite: bool = True) -> None:
        """Exports the plot into a chosen folder.

        Args:
            filename (str): The name of the figure.
            overwrite (bool): Chooses saving behaviour -> overwrite existing (if any) plots or always create a new one
        """
        version = 1
        file_extension = ".png"
        date = datetime.date.today().strftime("%d-%m-%Y")
        dest_folder = f"./reports/figures/{date}/"   # Should be changed to root path when testing is finished
        full_path = os.path.join(dest_folder, f"{filename}_ver_{version}{file_extension}")

        for i in range(0, 15): 
            if not os.path.exists(dest_folder) or overwrite:
                os.makedirs(dest_folder)
                plt.savefig(full_path, bbox_inches="tight")
            else:
                if i < 15:
                    version += 1
                else:
                    plt.savefig(full_path, bbox_inches="tight") # If 15 plots already exists, we overwrite the 15th and terminate loop

        print(f"Successfully exported '{filename+file_extension}'")

        
class LinePlot(Plot):
    """Subclass for plotting lineplot

    Args:
        Plot (Class): Inherits methods and attributes from Plot
    """
    def __init__(self, x, x_label, y_label, title, label):
        super().__init__(x_label, y_label, title)
        self.x = x
        self.label = label
        
    def lineplot(self) -> None:
        """Lineplot generator function
        """
        fig, ax = super().plot()
        
        # Input values
        for i in range(0, len(self.x)):
            ax.plot(
                self.x[i], 
                marker = 'o',
                markersize = 3,
                linewidth = 1.5,
                label = self.label[i])
        
        # Set legend
        fig.legend(
            loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes
        )
        
        plt.show() # Show plot - for testing
        

#x_data = np.array([[1, 3, 5], [3, 7, 5]])
#line_plot = LinePlot(x_data, "X", "Y", "Test titel", ["Test label", "Test label 2"])
#line_plot.lineplot()
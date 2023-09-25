import matplotlib.pyplot as plt
import datetime
import os

#from src.visualization.plot_settings import PlotSettings 

class Plot:
    """Parent class for plotting.
    
    Encapsulates all the necessary utilities needed in all plot subclasses.
    """
    def __init__(self, x_label, y_label, title):
        self.x_label = x_label
        self.y_label = y_label
        self.title = title
        
    def plot(self):
        # Create canvas
        fig, ax = plt.subplots()
        
        # Title and axis customization
        ax.set_title(self.title)
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        
        # Overall plot customization
        ax.grid(True)
        ax.tick_params(left=False, bottom=False, color="black")
        
        return fig, ax
        
    def _export_figure(self, filename: str, overwrite: bool = True) -> None:
        """Exports the plot into a folder, containing all figures.

        Args:
            filename (str): The name of the figure.
            overwrite (bool): Chooses saving behaviour -> overwrite existing (if any) plots or always create a new one
        """
        version = 1
        file_extension = ".png"
        date = datetime.date.today().strftime("%d-%m-%Y")
        path = "./figures/"   # Check path here
        full_path = os.path.join(path, f"{filename}_{date}_ver{version+file_extension}")

        for i in range(0, 15): 
            if not os.path.exists(full_path) or overwrite:
                plt.savefig(full_path, bbox_inches="tight")
            else:
                if i < 15:
                    version += 1
                else:
                    plt.savefig(full_path, bbox_inches="tight") # If 15 already exists, we overwrite the 15th

        print(f"Successfully exported '{filename+file_extension}'")

        
class LinePlot(Plot):
    """Subclass for plotting lineplot

    Args:
        Plot (Class): Inherits methods and attributes from Plot
    """
    def __init__(self, x, y, x_label, y_label, title, label):
        super().__init__(x_label, y_label, title)
        self.x = x
        self.y = y
        self.label = label
        
    def lineplot(self):
        fig, ax = super().plot()
        
        # Input values
        ax.plot(
            self.x,
            self.y, 
            marker = 'o',
            markersize = 3,
            linewidth = 1.5,
            label = self.label)
        
        plt.show() # Show plot - for testing
        
        
#x_data = [1, 3, 5]
#y_data = [3, 7, 5]

#line_plot = LinePlot(x_data, y_data, "X", "Y", "Test titel", "Test label")
#line_plot.lineplot()
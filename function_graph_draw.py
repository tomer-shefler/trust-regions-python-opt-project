import numpy as np
import matplotlib.pyplot as plt


class GraphDrawer:
    def __init__(self, func, graph_range=10):
        x = np.linspace(-graph_range, graph_range, 100)
        y = np.linspace(-graph_range, graph_range, 100)
        X, Y = np.meshgrid(x, y)
        Z, _, _ = func(np.array([X, Y]))
        fig = plt.figure()
        self.ax = fig.add_subplot(111, projection='3d')
        self.ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title(func.__name__)

    def draw_point(self, point):
        self.ax.scatter(point[0], point[1], point[2], color='black')

    def finish_draw(self):
        plt.show()
        
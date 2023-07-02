import numpy as np
import matplotlib.pyplot as plt
import os


class GraphDrawer:
    def __init__(self, func, graph_range=10):
        x = np.linspace(-graph_range, graph_range, 100)
        y = np.linspace(-graph_range, graph_range, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros(([len(x), len(y)]))

        for i in range(0, len(x)):
            for j in range(0, len(y)):
                Z[j, i], _, _ = func(np.array([x[i], y[j]]))
        fig = plt.figure()
        self.ax = fig.add_subplot(111, projection="3d")
        self.ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.7)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title(f"trust region - {func.__name__}")
        self.func = func

    def draw_point(self, point):
        self.ax.scatter(point[0], point[1], point[2], color="black")

    def finish_draw(self):
        plt.savefig(os.path.dirname(__file__) + f"/trust region - {self.func.__name__}")
        plt.show()

import numpy as np
import trust_region_dogleg
import function_graph_draw


if __name__ == "__main__":
    x0 = np.array([5,5])
    r = trust_region_dogleg.trust_region(x0)
    

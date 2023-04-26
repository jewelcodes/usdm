
# uSDM - minimalistic proof-of-concept species distribution modeling package
# this only exists for my own experimentation - not intended for actual research
# jewel, 2023

import logger
import matplotlib.pyplot as plt

def export(x, y, linecolor, title, filename):
    logger.debug("export_graph:", filename)
    plt.clf()
    plt.title(title)
    plt.plot(x, y, color = linecolor, linestyle = "solid")
    plt.title(title)
    plt.grid()
    plt.savefig(filename)

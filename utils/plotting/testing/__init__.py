import numpy as np

from utils.plotting import Plotter

plotter = Plotter()

plotter['mean'] += 1
plotter['mean'] += [1, 2, 3]
plotter['mean'].clear()
plotter['mean'] += np.array([1, 2, 3])

plotter['std'] += 1, 2, 3, 4, 5

x = np.arange(1, 10, 0.1)
y = np.sin(x)

plotter['sin'].data_(y, x)
plotter['sin'].name_('top kek')

plotter.show(sharex=True)

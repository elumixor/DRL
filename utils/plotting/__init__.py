from collections import Iterable

import matplotlib.pyplot as plt

import utils


class Plotter:
    class PlotterEntry:
        def __init__(self, name):
            self.name = name
            self.y = []
            self.__x = None

        def data_(self, y, x=None):
            self.y = y
            self.__x = x

        def name_(self, name):
            self.name = name

        @property
        def x(self):
            return list(range(len(self.y))) if self.__x is None else self.__x

        def __iadd__(self, value):
            if isinstance(value, Iterable):
                self.y += list(value)
            else:
                self.y.append(value)

        def clear(self):
            self.y = []

        def __iter__(self):
            return self.y.__iter__()

        def __str__(self):
            return str(self.y)

    def __init__(self):
        self.entries = dict()

    def __getitem__(self, name):
        entry = self.entries.get(name)
        if entry is None:
            entry = self.entries[name] = self.PlotterEntry(name)
        return entry

    def __setitem__(self, key, value):
        pass

    def __len__(self):
        return len(self.entries)

    def show(self, name=None, sharex=False, sharey=False, running_average=False):
        if name is None:
            if len(self) > 1:
                fig, axs = plt.subplots(len(self), sharex=sharex, sharey=sharey)
                # Plot all graphs
                for i, (name, entry) in enumerate(self.entries.items()):
                    axs[i].plot(entry.x, entry.y)
                    if running_average:
                        axs[i].plot(entry.x, utils.running_average(entry.y))
                    axs[i].set_title(entry.name)
            else:
                for name, entry in self.entries.items():
                    plt.plot(entry.y)
                    if running_average:
                        plt.plot(entry.x, utils.running_average(entry.y))
                    plt.title(entry.name)

            plt.show()
        else:
            entry = self.entries[name]
            data = entry.y
            plt.plot(data)
            if running_average:
                plt.plot(utils.running_average(data))
            plt.title(entry.name)
            plt.show()

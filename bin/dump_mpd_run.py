#!/usr/bin/env python

import sys
import itertools as it
import numpy as np

machine = sys.argv[1]
numbers = np.arange(*[int(n) for n in sys.argv[2].split("-")]).astype("str")
pyfile = sys.argv[3]

hosts = it.product([machine], numbers)
hosts = it.starmap(str.__add__, hosts)

with open("mpd.run", "w") as fp:
    fp.write("\n".join(hosts))
    fp.write("\n")

cmd = "mpiexec -n {} -f mpd.run python {}".format(len(numbers), pyfile)
print("Now run the cmd : \n", cmd)

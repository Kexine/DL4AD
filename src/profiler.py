#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import cProfile, pstats, io

import command_input

pr = cProfile.Profile()
pr.enable()

try:
    command_input.main()
except KeyboardInterrupt:
    pass

pr.disable()
s = io.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())

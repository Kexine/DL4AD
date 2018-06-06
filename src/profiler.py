#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import cProfile, pstats, io

# import command_input
from Extractor import H5Dataset
from customTransforms import *
from torchvision import transforms

pr = cProfile.Profile()

# dummy composition for debugging
composed = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,)),
                               ContrastNBrightness(1.5,0.5),
                               GaussianBlur(1.5),
                               SaltNPepper(0.1),
                               GaussianNoise(0, 0.1),
                               RegionDropout((10, 10),10)])

randomized = RandomApplyFromList([None],
                                 p=0.0,
                                 normalize=True,
                                 mandatory=[transforms.ToTensor(),
                                            transforms.Normalize((0.1,), (1.0,))])

train_set = H5Dataset(root_dir = '/home/flosmanm/data/AgentHuman/SeqTrain',
                      transform=randomized)


try:
    pr.enable()
    for i in range(len(train_set)):
        foo = train_set[i]
    # command_input.main()
except KeyboardInterrupt:
    pass

pr.disable()
s = io.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())

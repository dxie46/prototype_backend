import os

def label_func(x): return os.path.split(os.path.split(x)[0])[1]
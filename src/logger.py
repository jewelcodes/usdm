
# uSDM - minimalistic proof-of-concept species distribution modeling package
# this only exists for my own experimentation - not intended for actual research
# jewel, 2023

import sys, multiprocessing as mp
from datetime import datetime

last_progress = 0

# initialize logger
def load(f,v):
    global logfile
    global verbose
    global lock
    logfile = f
    verbose = v
    lock = mp.Lock()

# time
def logtime():
    t = datetime.now()
    return t.strftime("[%Y-%m-%d %H:%M:%S] ")

# verbose debug
def vdebug(*args):
    if verbose == 0:
        return

    global lock
    lock.acquire()

    global last_progress
    if last_progress != 0:
        print("\r", end = "")
        for i in range(last_progress):
            print(" ", end = "")

        print("\r", end = "")

    last_progress = 0

    print(logtime(), end = "")
    logfile.write(logtime())

    print("[verbose]", end = " ")
    logfile.write("[verbose] ")
    for arg in args:
        logfile.write(arg)
        logfile.write(" ")
        print(arg, end = " ")

    print("")
    logfile.write("\n")
    lock.release()

# normal debug
def debug(*args):
    global lock
    lock.acquire()

    global last_progress
    if last_progress != 0:
        print("\r", end = "")
        for i in range(last_progress):
            print(" ", end = "")

        print("\r", end = "")

    last_progress = 0

    print(logtime(), end = "")
    logfile.write(logtime())
    print("[debug]", end = " ")
    logfile.write("[debug] ")
    for arg in args:
        logfile.write(arg)
        logfile.write(" ")
        print(arg, end = " ")

    print("")
    logfile.write("\n")
    lock.release()

# show progress
def progress(*args):
    global lock
    lock.acquire()

    global last_progress

    print("\r", end="")
    print(logtime(), end = "")
    print("[progress] ", end="")

    s = 0

    for arg in args:
        s += len(arg) + 1
        print(arg, end = " ")

    print("    ", end="")

    s += 34     # timestamp + [progress]
    last_progress = s

    lock.release()

# error
def error(*args):
    global lock
    lock.acquire()

    global last_progress
    if last_progress != 0:
        print("\r", end = "")
        for i in range(last_progress):
            print(" ", end = "")

        print("\r", end = "")

    last_progress = 0

    print(logtime(), end = "")
    logfile.write(logtime())
    print("[error]", end = " ")
    logfile.write("[error] ")
    for arg in args:
        logfile.write(arg)
        logfile.write(" ")
        print(arg, end = " ")

    print("")
    logfile.write("\n")
    logfile.close()
    lock.release()
    sys.exit(-1)


#!/usr/bin/env python3

# uSDM - minimalistic proof-of-concept species distribution modeling package
# this only exists for my own experimentation - not intended for actual research
# jewel, 2023

import csv, sys, getopt
from datetime import datetime, timezone
import logger, records, env, raster, model

# overall program settings
verbose = 0
envdata = []
recdata = ""
threads = 1

def help(argv):
    print("usage:", argv[0], "-e [environmental data] -r [occurrence data]")
    exit(-1)

def main(argv):
    global verbose, envdata, recdata, threads

    try:
        opts, args = getopt.getopt(argv[1:], "hve:r:t:", ["help","verbose","env=","rec=","threads="])
    except getopt.GetoptError:
        help(argv)
    for opt, val in opts:
        if opt in("-h", "--help"):
            help(argv)
        elif opt in("-v", "--verbose"):
            verbose = 1
        elif opt in("-e", "--env"):
            envdata.append(val)
        elif opt in("-r", "--rec"):
            recdata = val
        elif opt in("-t", "--threads"):
            threads = int(val)
            if threads < 1 or threads > 8:
                print("error: threads must be between 1 and 8")
                exit(-1)

    if envdata == "" or recdata == "":
        help(argv)

    global logfile
    try:
        logfile = open("usdm.log", "w")
    except FileNotFoundError:
        print("could not open file usdm.log for writing")
        exit(-1)

    logger.load(logfile,verbose)

    logger.debug("starting")
    logger.vdebug("verbose mode enabled")
    logger.debug("using", str(threads), "threads for modeling")
    logger.debug("environmental data:")

    for envf in envdata:
        logger.debug(" +", envf)

    logger.debug("occurrence record data:")
    logger.debug(" +", recdata)

    records.load(recdata)
    env.init(envdata)

    model.work(threads, 1, envdata)

    logfile.close()
    exit(0)

if __name__ == "__main__":
    main(sys.argv[0:])

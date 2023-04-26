
# uSDM - minimalistic proof-of-concept species distribution modeling package
# this only exists for my own experimentation - not intended for actual research
# jewel, 2023

import logger, numpy as np, math
from collections import namedtuple

NOVALUE = -9999

# load a raster ASCII file
def load(file):
    try:
        h = open(file, "r")
    except FileNotFoundError:
        logger.error("unable to open file", file, "for reading")

    raster = namedtuple("raster", ["name", "w", "h", "xll", "yll", "xc", "yc", "xmax", "ymax", "cellsize", "cpd", "novalue", "raster", "maxval", "minval"])
    raster.name = file.split('/')[-1].split('.')[0]
    raster.novalue = "-9999"    # default
    raster.raster = []
    raster.maxval = NOVALUE
    raster.minval = -1*NOVALUE

    logger.vdebug("reading raster file", file)

    # read the file
    line_count = 0
    line = h.readline()
    cell_count = 0
    while line:
        # process header
        row = line.split(" ")
        if row[0].lower() in "ncols":
            #logger.vdebug("found ncols:", row[-1].split("\n")[0])
            raster.w = int(row[-1].split("\n")[0])
        elif row[0].lower() in "nrows":
            #logger.vdebug("found nrows:", row[-1].split("\n")[0])
            raster.h = int(row[-1].split("\n")[0])
        elif row[0].lower() in "xllcorner":
            raster.xll = float(row[-1].split("\n")[0])
        elif row[0].lower() in "yllcorner":
            raster.yll = float(row[-1].split("\n")[0])
        elif row[0].lower() in "cellsize":
            raster.cellsize = float(row[-1].split("\n")[0])
            raster.cpd = 1/raster.cellsize  # cells per degree
        elif row[0].lower() in "nodata_value":
            raster.novalue = float(row[-1].split("\n")[0])
        else:
            # actual data
            for data in row:
                if data != "" and data != " " and data != "\n":
                    val = float(data.split("\n")[0])
                    if val == raster.novalue:
                        val = NOVALUE
                    raster.raster.append(val)

                    if val != NOVALUE:
                        if val > raster.maxval:
                            raster.maxval = val
                        if val < raster.minval:
                            raster.minval = val

                    cell_count += 1

        line_count += 1

        old_percent = 7546  # will be rewritten

        if line_count >= 100:
            percent = math.floor((line_count*100)/raster.h)
            if percent != old_percent:
                logger.progress("reading raster " + raster.name + " " + str(percent) + "%")
                old_percent = percent

        line = h.readline()

    if cell_count > (raster.w * raster.h):
        logger.debug("WARNING: cell count of raster", raster.name, "does not match resolution; ignoring because it is larger")
        logger.debug("expected cell count:", str(raster.w*raster.h))
        logger.debug("actual cell count:", str(cell_count))
    elif cell_count < (raster.w * raster.h):
        logger.debug("WARNING: cell count of raster", raster.name, "does not match resolution; aborting because it is smaller")
        logger.debug("expected cell count:", str(raster.w*raster.h))
        logger.error("actual cell count:", str(cell_count))

    raster.xc = raster.xll
    raster.yc = raster.yll + (raster.h * raster.cellsize)

    raster.xmax = raster.xc + (raster.w * raster.cellsize)
    raster.ymax = raster.yc - (raster.h * raster.cellsize)

    logger.vdebug("read raster", raster.name, "width", str(raster.w), "height", str(raster.h), "cell size", str(raster.cellsize), "top-left", str(raster.xc), str(raster.yc), "boundaries", str(raster.xmax), str(raster.ymax), "min/max", str(raster.minval), str(raster.maxval))

    h.close()
    return raster

def load_header(file):
    try:
        h = open(file, "r")
    except FileNotFoundError:
        logger.error("unable to open file", file, "for reading")

    raster = namedtuple("raster", ["name", "w", "h", "xll", "yll", "xc", "yc", "xmax", "ymax", "cellsize", "cpd", "novalue", "raster", "maxval", "minval"])
    raster.name = file.split('/')[-1].split('.')[0]
    raster.novalue = "-9999"    # default
    raster.raster = []
    raster.maxval = NOVALUE
    raster.minval = -1*NOVALUE

    logger.vdebug("reading raster header from file", file)

    # read the file
    line_count = 0
    line = h.readline()
    cell_count = 0
    while line:
        # process header
        row = line.split(" ")
        if row[0].lower() in "ncols":
            #logger.vdebug("found ncols:", row[-1].split("\n")[0])
            raster.w = int(row[-1].split("\n")[0])
        elif row[0].lower() in "nrows":
            #logger.vdebug("found nrows:", row[-1].split("\n")[0])
            raster.h = int(row[-1].split("\n")[0])
        elif row[0].lower() in "xllcorner":
            raster.xll = float(row[-1].split("\n")[0])
        elif row[0].lower() in "yllcorner":
            raster.yll = float(row[-1].split("\n")[0])
        elif row[0].lower() in "cellsize":
            raster.cellsize = float(row[-1].split("\n")[0])
            raster.cpd = 1/raster.cellsize  # cells per degree
        elif row[0].lower() in "nodata_value":
            raster.novalue = float(row[-1].split("\n")[0])
        else:
            # actual data
            break

        line_count += 1

        if line_count >= 10:
            percent = math.floor((line_count*100)/raster.h)
            logger.progress("reading raster " + raster.name + " " + str(percent) + "%")

        line = h.readline()

    raster.xc = raster.xll
    raster.yc = raster.yll + (raster.h * raster.cellsize)

    raster.xmax = raster.xc + (raster.w * raster.cellsize)
    raster.ymax = raster.yc - (raster.h * raster.cellsize)

    logger.vdebug("read raster header", raster.name, "width", str(raster.w), "height", str(raster.h), "cell size", str(raster.cellsize), "top-left", str(raster.xc), str(raster.yc), "boundaries", str(raster.xmax), str(raster.ymax))

    h.close()
    return raster

def max(raster):
    #arr = np.array(raster.raster)
    #return np.max(arr)
    return raster.maxval

def min(raster):
    #arr = np.array(raster.raster)
    #part = np.partition(arr, arr.size-1)

    #i = 0
    #while i < part.size:
        #v = part[i]
        #if v != NOVALUE:
            #return v

        #i += 1

    return raster.minval

def read(raster, long, lat):
    if long >= raster.xmax or long < raster.xc or lat <= raster.ymax or lat > raster.yc:
        # out of bounds
        logger.vdebug("warning: attempt to read raster", raster.name, "out of bounds at", str(long), str(lat))
        return NOVALUE

    # cell position in degrees
    xdeg = long - raster.xc
    ydeg = raster.yc - lat

    #logger.debug("xdeg ", str(xdeg))
    #logger.debug("ydeg ", str(ydeg))
    #logger.debug("xc ", str(raster.xc))
    #logger.debug("yc ", str(raster.yc))
    #logger.debug("cpd ", str(raster.cpd))

    # cell position in cell units
    xcell = math.floor(xdeg * raster.cpd)
    ycell = math.floor(ydeg * raster.cpd)

    #logger.debug("read ", str(long), str(lat), "at", str(xcell), str(ycell))
    #logger.error("value ", str(raster.raster[(ycell*raster.w)+xcell]))

    #logger.vdebug("read raster", raster.name, "position", str(long), str(lat), "is at cell", str(xcell), str(ycell))
    return raster.raster[((ycell*raster.w)+xcell)]

def read_line(raster, lat):
    if lat <= raster.ymax or lat > raster.yc:
        logger.vdebug("warning: attempt to read raster", raster.name, "out of bounds at", str(long), str(lat))
        return NOVALUE
    
    # position in degrees
    ydeg = raster.yc - lat
    ycell = math.floor(ydeg * raster.cpd)

    pos = ycell*raster.w

    return raster.raster[pos:pos+raster.w]

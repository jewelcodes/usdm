
# uSDM - minimalistic proof-of-concept species distribution modeling package
# this only exists for my own experimentation - not intended for actual research
# jewel, 2023

import logger, raster

count = 0
loaded = 0
width = 0
height = 0
xc = 0
yc = 0
xll = 0
yll = 0
cellsize = 0
cell_count = 0
names = []

def init(envfiles):
    global count, names
    global loaded, width, height, xc, yc, cellsize, cell_count, xll, yll
    for f in envfiles:
        names.append(f.split('/')[-1].split('.')[0])

        factor = raster.load_header(f)
        if count == 0:
            width = factor.w
            height = factor.h
            xc = factor.xc
            yc = factor.yc
            xll = factor.xll
            yll = factor.yll
            cellsize = factor.cellsize
            cell_count = factor.w * factor.h
        else:
            if factor.w != width or factor.h != height or factor.xc != xc or factor.yc != yc or factor.cellsize != cellsize:
                logger.error("mismatch in raster resolution of environmental factor", factor.name)

        count += 1

def load(file):
    global loaded, width, height, xc, yc, cellsize, cell_count, xll, yll
    factor = raster.load(file)

    loaded += 1
    return factor

    # test
    #raster.read(factors[0], 8.544, 61.664)

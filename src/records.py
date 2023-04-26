
# uSDM - minimalistic proof-of-concept species distribution modeling package
# this only exists for my own experimentation - not intended for actual research
# jewel, 2023

import logger, csv

# loads species occurrence records from a csv file
def load(file):
    global long, lat, species_name, count
    long = []
    lat = []
    species_name = ""
    count = 0

    try:
        h = open(file, "r")
    except FileNotFoundError:
        logger.error("unable to open file", file, "for reading")
    
    csvfile = csv.reader(h, delimiter = ',')
    line = 0
    coordinate_order = 0    # 0 = lat,long, 1 = long,lat

    # first row is header
    for row in csvfile:
        if line == 0:
            if row[1] in("long","longitude","decimalLongitude","decimalLong"):
                coordinateOrder = 1
                logger.debug("using column", row[1], "as longitude and", row[2], "as latitude")
            else:
                coordinateOrder = 0
                logger.debug("using column", row[1], "as latitude and", row[2], "as longitude")
        elif line == 1:
            species_name = row[0]
        else:
            if species_name != row[0]:
                logger.error("species name mismatch at line", str(line))

        if line >= 1:
            # save the occurrences
            if row[1] == "" or row[2] == "":
                logger.debug("warning: ignoring empty occurrence at line", str(line))
            else:
                if coordinateOrder == 1:
                    long.append(float(row[1]))
                    lat.append(float(row[2]))
                else:
                    lat.append(float(row[1]))
                    long.append(float(row[2]))

            #logger.vdebug("species occurrence record at", str(long[-1]), str(lat[-1]))
        
        line += 1

    count = line - 1;    
    logger.debug("loaded species", species_name, "with", str(count), "occurrence records")
    h.close()

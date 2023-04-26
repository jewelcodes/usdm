
# uSDM - minimalistic proof-of-concept species distribution modeling package
# this only exists for my own experimentation - not intended for actual research
# jewel, 2023

import logger, raster, env, records, graph, multiprocessing as mp
import time, ctypes
import os, numpy as np, math
from scipy.stats import norm
from datetime import datetime

graph_x = []
graph_y = []
verbose_thread = 0

# for frequency distributions
env_class_multiple = 2        # feature range * this value
minimum_classes = 10

def work(threads, iterations, envfiles):
    global dest_path
    dest_path = records.species_name
    try:
        os.mkdir(dest_path)
    except FileExistsError:
        logger.debug("warning: folder already exists, ignoring")

    global graphs_path
    graphs_path = dest_path + "/graphs"

    try:
        os.mkdir(graphs_path)
    except FileExistsError:
        logger.debug("warning: folder already exists, ignoring")
        
    i = 0
    while i < iterations:
        model(threads, i, envfiles)
        i += 1

def model_thread(lock, tid, iteration, envfiles, start_factor, count):
    global sample_mean, sample_stddev, sample_median, sample_min, sample_max, env_weight
    global model_prob_acc_arith
    global graph_x, graph_y, thread_progress, thread_factors, graph_count
    global env_min, env_max, env_ll, env_hl
    global env_class_multiple

    sec = datetime.now().second
    old_sec = sec

    for factor_no in range(start_factor, count+start_factor):
        thread_factors[tid] = factor_no
        factor = env.load(envfiles[factor_no])
        logger.debug("thread", str(tid), "is modeling", records.species_name, "on", factor.name)

        env_min[factor_no] = raster.min(factor)
        env_max[factor_no] = raster.max(factor)
        env_range = env_max[factor_no]-env_min[factor_no]

        logger.vdebug("feature min/max for factor", factor.name, "is", str(env_min[factor_no]) + "/" + str(env_max[factor_no]))
        logger.vdebug("feature max for factor", factor.name, "is", str(env_max[factor_no]))
        logger.vdebug("feature range for factor", factor.name, "is", str(env_range))
        #logger.vdebug("sample mean for factor", factor.name, "is", str(sample_mean[factor_no]))
        #logger.vdebug("sample stddev for factor", factor.name, "is", str(sample_stddev[factor_no]))
        logger.vdebug("sample median for factor", factor.name, "is", str(sample_median[factor_no]))
        logger.vdebug("sample 10th/90th percentiles for factor", factor.name, "is", str(sample_min[factor_no]) + "/" + str(sample_max[factor_no]))
        logger.vdebug("weight of factor", factor.name, "is", str(env_weight[factor_no]))

        sample_classes = math.floor(env_range*env_class_multiple)
        if sample_classes < minimum_classes:
            sample_classes = minimum_classes
        elif sample_classes & 1:
            sample_classes += 1
        sample_class_w = env_range/sample_classes

        sample_frequency = []
        sample_class = []

        for i in range(sample_classes):
            sample_frequency.append(0)
            class_min = (i*sample_class_w)+env_min[factor_no]
            class_max = class_min + sample_class_w
            class_mean = (class_min+class_max)/2
            sample_class.append(class_mean)

        for occ in range(records.count):
            val = raster.read(factor, records.long[occ], records.lat[occ])
            if val != raster.NOVALUE:
                # determine this value's class
                c = math.floor((val-env_min[factor_no])/sample_class_w)
                if c >= sample_classes:
                    c = sample_classes-1
                sample_frequency[c] += 1

        # to 0-1 range
        highest_freq = 0
        for i in range(sample_classes):
            if sample_frequency[i] > highest_freq:
                highest_freq = sample_frequency[i]

        for i in range(sample_classes):
            sample_frequency[i] /= highest_freq
            sample_frequency[i] *= env_weight[factor_no]

        # now construct the actual model
        model_raster = open(dest_path + "/" + records.species_name + "_" + str(iteration) + "_" + factor.name + ".asc", "w")

        model_raster.write("ncols " + str(factor.w) + "\n")
        model_raster.write("nrows " + str(factor.h) + "\n")
        model_raster.write("cellsize " + str(factor.cellsize) + "\n")
        model_raster.write("yllcorner " + str(factor.yll) + "\n")
        model_raster.write("xllcorner " + str(factor.xll) + "\n")
        model_raster.write("nodata_value " + str(raster.NOVALUE) + "\n")

        long = factor.xc
        lat = factor.yc

        # this part can take a long time so show progress per latitude
        row = 0
        old_percent = 5489  # will be rewritten
        norm_zero = norm.pdf(0)

        cell = 0    # current cell count

        while lat > factor.ymax:
            line = raster.read_line(factor, lat)

            #lock.acquire()
            if line == raster.NOVALUE:
                for i in range(factor.w):
                    model_raster.write(str(raster.NOVALUE) + " ")
                    model_prob_acc_arith[cell] = raster.NOVALUE
                    cell += 1
            else:
                # valid row
                for i in range(factor.w):
                    env_val = line[i]
                    if env_val == raster.NOVALUE:
                        model_raster.write(str(raster.NOVALUE) + " ")
                        model_prob_acc_arith[cell] = raster.NOVALUE
                        cell += 1
                    else:
                        #z = (env_val - sample_mean[factor_no]) / sample_stddev[factor_no]
                        #prob = norm.pdf(z) / norm_zero
                        # calculate probability according to frequency distribution
                        c = math.floor((env_val-env_min[factor_no])/sample_class_w)
                        if c >= sample_classes:
                            c = sample_classes-1
                        prob = sample_frequency[c]
                        model_raster.write(str(prob) + " ")
                        
                        model_prob_acc_arith[cell] += prob

                        cell += 1

            #lock.release()
            model_raster.write("\n")
            lat -= factor.cellsize
            long = factor.xc
            row += 1
            
            sec = datetime.now().second

            if sec != old_sec:
                percent = math.floor((row*100)/factor.h)
                thread_progress[tid] = percent

                if tid == verbose_thread:              
                    old_sec = sec
                    progress = "iteration " + str(iteration) + " on "

                    for t in range(thread_count):
                        progress += env.names[thread_factors[t]] + " " + str(thread_progress[t]) + "%"
                        if t != thread_count-1:
                            progress += ", "

                    logger.progress(progress)

        logger.debug("completed iteration", str(iteration), "on factor", factor.name)

        # done for this factor
        model_raster.close()

def train(envfiles):
    global sample_mean, sample_stddev, sample_median, sample_min, sample_max, env_weight
    global graph_x, graph_y, thread_progress, thread_factors, graph_count, range_ratio
    global env_min, env_max, env_ll, env_hl
    global env_class_multiple

    for factor_no in range(env.count):
        factor = env.load(envfiles[factor_no])

        global env_store
        env_store = []   # dump all values here to calculate mean and stddev

        sample_min[factor_no] = -1*raster.NOVALUE
        sample_max[factor_no] = raster.NOVALUE
        env_min[factor_no] = raster.min(factor)
        env_max[factor_no] = raster.max(factor)
        env_range = env_max[factor_no]-env_min[factor_no]

        for occ in range(records.count):
            val = raster.read(factor, records.long[occ], records.lat[occ])
            if val == raster.NOVALUE:
                logger.debug("SKIP: no data available for", factor.name, "at", str(records.long[occ]), str(records.lat[occ]))
            else:
                env_store.append(val)
                if val > sample_max[factor_no]:
                    sample_max[factor_no] = val
                elif val < sample_min[factor_no]:
                    sample_min[factor_no] = val
                #logger.debug("value", str(val))

        sample_mean[factor_no] = np.mean(np.array(env_store))
        sample_stddev[factor_no] = np.std(np.array(env_store))
        sample_median[factor_no] = np.median(np.array(env_store))
        env_min[factor_no] = raster.min(factor)
        env_max[factor_no] = raster.max(factor)
        env_range = env_max[factor_no]-env_min[factor_no]

        # construct frequency distribution
        sample_classes = math.floor(env_range*env_class_multiple)
        if sample_classes < minimum_classes:
            sample_classes = minimum_classes
        elif sample_classes & 1:
            sample_classes += 1
        sample_class_w = env_range/sample_classes
        logger.vdebug("using", str(sample_classes), "classes for factor", factor.name, "class width", str(sample_class_w))

        sample_frequency = []
        sample_class = []

        # these will be replaced
        sample_lowest_class = sample_classes*2
        sample_highest_class = -1*sample_lowest_class

        # sample frequency distribution
        for i in range(sample_classes):
            sample_frequency.append(0)
            class_min = (i*sample_class_w)+env_min[factor_no]
            class_max = class_min + sample_class_w
            class_mean = (class_min+class_max)/2
            sample_class.append(class_mean)

        for occ in range(records.count):
            val = raster.read(factor, records.long[occ], records.lat[occ])
            if val != raster.NOVALUE:
                # determine this value's class
                c = math.floor((val-env_min[factor_no])/sample_class_w)
                if c >= sample_classes:
                    c = sample_classes-1

                sample_frequency[c] += 1
                if c < sample_lowest_class:
                    sample_lowest_class = c
                elif c > sample_highest_class:
                    sample_highest_class = c

        sample_frequency_smooth = []
        sample_class_smooth = []

        for i in range(int(sample_classes/2)):
            sample_class_smooth.append((sample_class[i*2]+sample_class[(i*2)+1])/2)
            sample_frequency_smooth.append((sample_frequency[i*2]+sample_frequency[(i*2)+1])/2)

        # to 0-1 range
        highest_freq = 0
        for i in range(len(sample_class_smooth)):
            if sample_frequency_smooth[i] > highest_freq:
                highest_freq = sample_frequency_smooth[i]

        for i in range(len(sample_class_smooth)):
            sample_frequency_smooth[i] /= highest_freq

        graph.export(sample_class_smooth, sample_frequency_smooth, "#FF0000", "Distribution of " + records.species_name + " according to " + factor.name, graphs_path + "/" + records.species_name + "_" + factor.name + ".png")

        # environmental frequency distribution
        env_frequency = []
        for i in range(sample_classes):
            env_frequency.append(0)

        # TODO: read this from files
        freq_file = open(envfiles[factor_no] + "_freq.csv", "w")
        freq_file.write("class,ll,hl,f,cf\n")

        lat = factor.yc
        row = 0
        old_percent = 758
        old_sec = datetime.now().second

        while lat > factor.ymax:
            line = raster.read_line(factor, lat)

            if line != raster.NOVALUE:
                for i in range(factor.w):
                    val = line[i]
                    if val != raster.NOVALUE:
                        c = math.floor((val-env_min[factor_no])/sample_class_w)
                        if c >= sample_classes:
                            c = sample_classes-1

                        env_frequency[c] += 1

            lat -= factor.cellsize
            row += 1

            sec = datetime.now().second
            if sec != old_sec:
                percent = math.floor((row*100)/factor.h)

                if percent != old_percent:
                    old_percent = percent

                    logger.progress("frequency distribution for feature", factor.name, str(percent) + "%")

        # feature cumulative frequency distribution
        env_cf = []
        env_cf.append(env_frequency[0])

        for i in range(1, sample_classes):
            env_cf.append(env_cf[i-1] + env_frequency[i])

        # write the frequency distribution to speed up subsequent runs
        for i in range(sample_classes):
            # class,ll,hl,f,cf
            freq_file.write(str(i+1) + ",")
            freq_file.write(str(env_min[factor_no]+(i*sample_class_w)) + ",")
            freq_file.write(str(env_min[factor_no]+(i+1)*sample_class_w) + ",")
            freq_file.write(str(env_frequency[i]) + ",")
            freq_file.write(str(env_cf[i]) + "\n")

        freq_file.close()

       # now calculate 2.5th and 97.5th percentiles of the feature itself
        freq_10p = env_cf[-1] * 0.025
        freq_90p = env_cf[-1] * 0.975

        class_10p = 0
        class_90p = 0

        for i in range(sample_classes):
            if i == 0:
                low = 0
            else:
                low = env_cf[i-1]
            
            high = env_cf[i]

            if freq_10p >= low and freq_10p < high:
                class_10p = i
            
            if freq_90p >= low and freq_90p < high:
                class_90p = i

        env_ll[factor_no] = sample_class[class_10p]
        env_hl[factor_no] = sample_class[class_90p]

        # write this too into a file for better performance next run
        stat_file = open(envfiles[factor_no] + "_stat.csv", "w")
        stat_file.write("name," + factor.name + "\n")
        stat_file.write("min," + str(env_min[factor_no]) + "\n")
        stat_file.write("max," + str(env_max[factor_no]) + "\n")
        stat_file.write("10th percentile," + str(env_ll[factor_no]) + "\n")
        stat_file.write("90th percentile," + str(env_hl[factor_no]) + "\n")
        stat_file.close()

        # sample cumulative frequency distribution
        sample_cf = []
        sample_cf.append(sample_frequency[0])

        for i in range(1, sample_classes):
            sample_cf.append(sample_cf[i-1] + sample_frequency[i])

        # class of 10th and 90th percentilee
        freq_10p = sample_cf[-1] * 0.1
        freq_90p = sample_cf[-1] * 0.9

        class_10p = 0
        class_90p = 0

        for i in range(sample_classes):
            if i == 0:
                low = 0
            else:
                low = sample_cf[i-1]
            
            high = sample_cf[i]

            if freq_10p >= low and freq_10p < high:
                class_10p = i
            
            if freq_90p >= low and freq_90p < high:
                class_90p = i

        #logger.debug("10th percentile class is", str(class_10p))
        #logger.debug("90th percentile class is", str(class_90p))

        sample_min[factor_no] = sample_class[class_10p]
        sample_max[factor_no] = sample_class[class_90p]

        if (sample_min[factor_no] > env_ll[factor_no]) and (sample_max[factor_no] < env_hl[factor_no]):
            env_min[factor_no] = env_ll[factor_no]
            env_max[factor_no] = env_hl[factor_no]
            logger.vdebug("eliminated outliers of feature layer", factor.name)

        #logger.debug("species 10/90 is", str(sample_min[factor_no]), str(sample_max[factor_no]))
        #logger.debug("feature 10/90 is", str(env_ll[factor_no]), str(env_hl[factor_no]))

        range_ratio[factor_no] = (sample_max[factor_no]-sample_min[factor_no])/(env_max[factor_no]-env_min[factor_no])

    #logger.vdebug("sample 10th/90th percentiles for factor", factor.name, "is", str(sample_min[factor_no]) + "/" + str(sample_max[factor_no]))

    # now they're all loaded so we can weigh them
    #
    # uncalibrated weight = sum range ratio / range ratio
    # final weight = uncalibrated weight / sum uncalibrated weights
    # this gives us half weight for a feature with 2x range relative to another

    total_range_ratio = 0

    for i in range(env.count):
        total_range_ratio += range_ratio[i]

    uncal_weight = []
    total_uncal_weight = 0

    for i in range(env.count):
        uncal_weight.append(total_range_ratio/range_ratio[i])
        total_uncal_weight += (total_range_ratio/range_ratio[i])

    for i in range(env.count):
        env_weight[i] = uncal_weight[i]/total_uncal_weight

def model(threads, iteration, envfiles, *args):
    logger.debug("starting iteration", str(iteration))

    global env_file
    env_file = open(dest_path + "/" + records.species_name + "_" + str(iteration) + "_env.csv", "w")

    #env_file.write("feature,min,max,range,sample min,sample max,sample range,range %,significance %,sample median,sample mean,sample stddev,stddev % of mean\n")

    env_file.write("feature,outliers,min,max,range,sample p10,sample p90,sample range,range %,sample median,sample mean,sample stddev,weight\n")
    
    global sample_mean, sample_stddev, sample_median, sample_min, sample_max
    global model_prob_acc_arith
    global graph_x, graph_y, graph_count, thread_progress, thread_factors
    global range_ratio
    global env_min, env_max, env_weight
    global env_ll, env_hl

    cell_count = env.width * (env.height+2)

    sample_mean = mp.RawArray(ctypes.c_double, env.count)
    sample_stddev = mp.RawArray(ctypes.c_double, env.count)
    sample_median = mp.RawArray(ctypes.c_double, env.count)
    sample_min = mp.RawArray(ctypes.c_double, env.count)
    sample_max = mp.RawArray(ctypes.c_double, env.count)
    env_min = mp.RawArray(ctypes.c_double, env.count)
    env_max = mp.RawArray(ctypes.c_double, env.count)
    env_ll = mp.RawArray(ctypes.c_double, env.count)
    env_hl = mp.RawArray(ctypes.c_double, env.count)
    graph_x = mp.RawArray(ctypes.POINTER(ctypes.c_double), env.count)
    graph_y = mp.RawArray(ctypes.POINTER(ctypes.c_double), env.count)
    graph_count = mp.RawArray(ctypes.c_int, env.count)
    range_ratio = mp.RawArray(ctypes.c_double, env.count)
    env_weight = mp.RawArray(ctypes.c_double, env.count)
    model_prob_acc_arith = mp.Array(ctypes.c_double, cell_count) # accumulator for arithmetic mean
    thread_progress = mp.RawArray(ctypes.c_int, threads)
    thread_factors = mp.RawArray(ctypes.c_int, threads)

    for i in range(env.count):
        sample_mean[i] = 0
        sample_median[i] = 0
        sample_stddev[i] = 0
        sample_min[i] = 0
        sample_max[i] = 0
        env_min[i] = 0
        env_max[i] = 0
        graph_count[i] = 0
        range_ratio[i] = 0
        env_weight[i] = 0

    t = []

    # construct f distributions and eliminate outliers that will be used for the actual model
    train(envfiles)

    for i in range(env.count):
        # feature,outliers,min,max,range,10th percentile,90th percentile,sample range,range %,sample median,sample mean,sample stddev,weight
        env_file.write(env.names[i] + ",")                                  # feature

        if env_min[i] == env_ll[i]:                                        # outliers
            env_file.write("no,")
        else:
            env_file.write("yes,")

        env_file.write(str(env_min[i]) + ",")                               # min
        env_file.write(str(env_max[i]) + ",")                               # max
        env_file.write(str(env_max[i]-env_min[i]) + ",")                    # range
        env_file.write(str(sample_min[i]) + ",")                       # sample p10
        env_file.write(str(sample_max[i]) + ",")                       # sample p90
        env_file.write(str(sample_max[i]-sample_min[i]) + ",")    # sample range
        env_file.write(str(range_ratio[i]*100) + "%,")                      # range %
        env_file.write(str(sample_median[i]) + ",")                            # sample median
        env_file.write(str(sample_mean[i]) + ",")                              # sample mean
        env_file.write(str(sample_stddev[i]) + ",")                            # sample stddev
        env_file.write(str(env_weight[i]*100) + "%\n")          # weight

    env_file.close()

    global thread_count, env_lock
    lock = mp.Lock()
    env_lock = mp.Lock()

    if threads == 1:
        thread_count = 1
        model_thread(lock, 0, iteration, envfiles, 0, env.count)
    elif threads >= env.count:
        thread_count = env.count

        for i in range(env.count):
            logger.vdebug("thread", str(i), "used to model factor", str(i))
            t.append(mp.Process(target=model_thread, args=(lock, i, iteration, envfiles, i, 1,)))

        #t[0].start()
        #t[0].join()
        
        for i in range(env.count):
            t[i].start()

        for i in range(env.count):
            t[i].join()
    else:
        thread_count = threads

        factors_per_thread = math.ceil(env.count/thread_count)

        f = 0

        logger.vdebug("total of", str(env.count), "factors")

        t_setup = 0
        f_setup = 0
        last_end = -1

        for i in range(thread_count):
            t_setup += 1
            start = last_end + 1
            c = factors_per_thread
            end = start + c - 1

            while end >= env.count:
                c -= 1
                if c == 0:
                    logger.error("unable to divide tasks among processes")

                end = start + c - 1

            last_end = end

            if start != end:
                logger.vdebug("thread", str(i), "used to model factors from", str(start), "to", str(end), "count", str(c))
            else:
                logger.vdebug("thread", str(i), "used to model factor", str(start), "count", str(c))

            t.append(mp.Process(target=model_thread, args=(lock, i, iteration, envfiles, start, c,)))

        for i in range(thread_count):
            t[i].start()

        for i in range(thread_count):
            t[i].join()

    # done for all factors, now calculate the mean
    logger.debug("finalizing model for iteration", str(iteration))

    # arithmetic mean
    #model_prob_arith = np.array(model_prob_acc_arith)/env.count
    #model_prob_final = model_prob_arith*env.count

    model_prob_final = model_prob_acc_arith     # already weighted, there's nothing to do

    model_raster = open(dest_path + "/" + records.species_name + "_" + str(iteration) + ".asc", "w")

    model_raster.write("ncols " + str(env.width) + "\n")
    model_raster.write("nrows " + str(env.height) + "\n")
    model_raster.write("cellsize " + str(env.cellsize) + "\n")
    model_raster.write("yllcorner " + str(env.yll) + "\n")
    model_raster.write("xllcorner " + str(env.xll) + "\n")
    model_raster.write("nodata_value " + str(raster.NOVALUE) + "\n")

    cell = 0
    old_sec = datetime.now().second
    for y in range(env.height):
        for x in range(env.width):
            if model_prob_final[cell] < 0 or model_prob_final[cell] > 1:
                model_raster.write(str(raster.NOVALUE) + " ")
            else:
                model_raster.write(str(model_prob_final[cell]) + " ")

            cell += 1

        model_raster.write("\n")

        sec = datetime.now().second

        if sec != old_sec:
            old_sec = sec

            percent = math.floor(cell/cell_count)

            logger.progress("writing", dest_path + "/" + records.species_name + "_" + str(iteration) + ".asc", str(percent) + "%")

    model_raster.close()

    test_file = open(dest_path + "/" + records.species_name + "_" + str(iteration) + "_samples.csv", "w")

    logger.debug("calculating error rate for iteration", str(iteration))

    test_file.write("longitude,latitude,prediction,error\n")

    # now reload the raster for testing
    model_raster = raster.load(dest_path + "/" + records.species_name + "_" + str(iteration) + ".asc")
    error_sq = 0
    valid_occ = 0

    for occ in range(records.count):
        val = raster.read(model_raster, records.long[occ], records.lat[occ])
        if val != raster.NOVALUE:
            valid_occ += 1

            error = 1 - val
            test_file.write(str(records.long[occ]) + ",")
            test_file.write(str(records.lat[occ]) + ",")
            test_file.write(str(val) + ",")
            test_file.write(str(error) + "\n")

            error_sq += np.power(error, 2)
        
    test_file.close()

    logger.debug("mean squared error", str(error_sq/valid_occ))

    # now put the final results file
    final_file = open(dest_path + "/" + records.species_name + "_" + str(iteration) + ".csv", "w")
    final_file.write("species name," + records.species_name + "\n")
    final_file.write("total records," + str(records.count) + "\n")
    final_file.write("valid records," + str(valid_occ) + "\n")
    final_file.write("environmental factors," + str(env.count) + "\n")
    final_file.write("mean squared error," + str(error_sq/valid_occ) + "\n")
    final_file.close()

    logger.debug("completed iteration", str(iteration))

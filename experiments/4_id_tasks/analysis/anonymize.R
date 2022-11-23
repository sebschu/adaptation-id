#!/usr/bin/env Rscript

library(stringr)
library(plyr)

args = commandArgs(trailingOnly=TRUE)


d = read.csv(args[1])
m = data.frame()
for (i in 2:length(args)) {
  m = rbind(m, read.csv(args[i]))
}


out_fname = str_replace(args[1], ".csv", "-anon.csv")

d_new = d

d_new$workerid = mapvalues(d$workerid, m$prolific_participant_id, m$workerid)

write.csv(d_new, out_fname)


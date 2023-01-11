library(dplyr)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))


df <- read.csv('../../data/ktt_parsed.csv')

scores <- df %>% group_by(workerid) %>% summarize(score = sum(trialScore))

write.csv(scores, 'scores_ktt.csv', row.names=FALSE)

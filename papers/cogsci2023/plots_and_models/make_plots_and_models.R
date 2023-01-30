
library(lme4)
library(lmerTest)
library(tidyverse)
library(effectsize)
library(texreg)

theme_set(theme_bw())

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# load utility functions
source("./helpers.R")

#########################################
# Reanalysis of Schuster & Degen (2019) #
#########################################
# Load data

trials = remove_quotes(format_data(read.csv("../../../../adaptation/experiments/11_talker_specific_adaptation_fixed/data/11_talker_specific_adaptation_fixed-trials.csv")))
exp_trials = remove_quotes(read.csv("../../../../adaptation/experiments/11_talker_specific_adaptation_fixed//data/11_talker_specific_adaptation_fixed-exp_trials.csv"))
trials[, c("test_order", "first_speaker_type", "confident_speaker")] = str_split(trials$condition, "_", simplify=T)

# Exclusions

get_correct_catch_trial_counts = function (data) {
  ret = data %>% 
    filter(., catch_trial == 1) %>%
    group_by(workerid) %>%
    summarise(catch_perf = sum(catch_trial_answer_correct), catch_prop = sum(catch_trial_answer_correct)/n())
  
  return(ret)
}

EXCLUDE_BELOW = 11

catch_trial_perf = get_correct_catch_trial_counts(exp_trials)

exclude = catch_trial_perf %>%
  filter(catch_perf < EXCLUDE_BELOW) %>%
  .$workerid

print(paste("Excluded", length(exclude), "participants based on catch-trial performance in S&D (2019) Exp 1 data."))
d = trials %>% filter(., !(workerid %in% exclude))

d_overall_means = d %>%
  group_by(modal, workerid) %>% 
  summarise(rating_m_overall = mean(rating))

d_indiv_means =  d %>%
  group_by(modal,percentage_blue, workerid) %>% 
  summarise(rating_m = mean(rating))


d_indiv_merged = merge(d_indiv_means, d_overall_means, by=c("workerid", "modal"))

cors = d_indiv_merged %>%
  group_by(workerid) %>%
  summarise(corr = cor(rating_m, rating_m_overall))


exclude = cors %>%
  filter(corr > 0.75) %>%
  .$workerid

print(paste("Excluded", length(exclude), "participants based on random responses in S&D (2019) Exp 1 data."))
d = d %>% filter(!(workerid %in% exclude))

# compute AUCs

#AUCs for cautious speaker ratings 
aucs.cautious = d %>% filter(speaker_type == "cautious") %>% auc_for_participants(., method=auc_method2)

#AUCs for confident speaker ratings 
aucs.confident = d %>% filter(speaker_type == "confident") %>% auc_for_participants(., method=auc_method2)

aucs.cautious$cond = "cautious (might-biased)"

aucs.confident$cond = "confident (probably-biased)"

aucs.all = rbind(aucs.cautious, aucs.confident)


# Load model likelihoods
d_likelihoods.cautious = read.csv("../../../../adaptation/models/4_individual_differences/model-runs/theta-prior/cautious/likelihood")

d_likelihoods.confident = read.csv("../../../../adaptation/models/4_individual_differences/model-runs/theta-prior/confident/likelihood")

d_likelihoods.merged = merge(d_likelihoods.cautious, 
                             d_likelihoods.confident,
                             by=c("workerid", "condition"))
d_likelihoods.merged = d_likelihoods.merged %>% mutate(most_likely_model = case_when(likelihood.x > likelihood.y ~ "cautious", TRUE ~ "confident" ) )

d.all = merge(d_likelihoods.merged, d %>% 
                select(-condition) %>% 
                group_by(workerid, speaker_type, test_order, confident_speaker) %>% 
                summarise(first_speaker_type = first(first_speaker_type)) %>% 
                rename(condition = speaker_type), 
              by=c("workerid", "condition"))


d.all = d.all %>% 
  rename(likelihood.cautious = likelihood.x, 
         likelihood.confident = likelihood.y) 

d.lr = d.all %>% 
  pivot_longer(starts_with("likelihood."), 
               names_prefix="likelihood.", 
               names_to="likelihood") %>% 
  group_by(workerid, condition, most_likely_model, confident_speaker, test_order, first_speaker_type) %>%
  summarise(likelihood_ratio = diff(value)) %>%
  ungroup() %>%
  select(workerid, condition, likelihood_ratio, most_likely_model, first_speaker_type, test_order,confident_speaker)


d.diff = d.lr %>% 
  dplyr::filter(condition == "cautious") %>%
  dplyr::rename(likelihood_ratio.cautious = likelihood_ratio) %>%
  dplyr::rename(most_likely_model.cautious = most_likely_model) %>%
  dplyr::select(workerid,  likelihood_ratio.cautious, most_likely_model.cautious, first_speaker_type, test_order, confident_speaker) %>%
  merge(d.lr %>% 
          dplyr::filter(condition == "confident") %>%
          dplyr::rename(likelihood_ratio.confident = likelihood_ratio) %>%
          dplyr::rename(most_likely_model.confident = most_likely_model) %>% 
          dplyr::select(workerid, likelihood_ratio.confident, most_likely_model.confident)) %>%
  mutate(likelihood_diff = likelihood_ratio.confident - likelihood_ratio.cautious) %>%
  mutate(most_likely_model.cautious.correct = most_likely_model.cautious == "cautious") %>%
  mutate(most_likely_model.confident.correct = most_likely_model.confident == "confident") %>%
  mutate(adapter_type = case_when(
    most_likely_model.confident.correct & most_likely_model.cautious.correct ~ "adapted to both speakers",
    most_likely_model.confident.correct & !most_likely_model.cautious.correct ~ "always confident",
    !most_likely_model.confident.correct & most_likely_model.cautious.correct ~ "always cautious",
    TRUE ~ "adapted to wrong speakers"
  ))



reanalysis_plot = d.diff %>% 
  mutate(adapter_type = factor(adapter_type, levels=c("adapted to wrong speakers", "always cautious", "always confident", "adapted to both speakers"))) %>%
  ggplot(aes(x=likelihood_diff, col=adapter_type, pch=adapter_type)) + 
  geom_jitter(aes(y=0),  height=0.3, size=2) + ylim(-1,1) + 
  geom_hline(yintercept=0) + 
  theme(axis.ticks.y = element_blank(), 
        axis.title.y = element_blank(), 
        axis.text.y = element_blank(), 
        legend.position = "bottom", 
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()) +
  geom_linerange(x=0, ymin=-.3, ymax=.3, col="black") +
  xlab ("Difference in likelihood ratio across two test conditions\n(LLR confident - LLR cautious)") + 
  guides(pch=guide_legend(title="Adapter type"), col=guide_legend("Adapter type"))

ggsave(filename="reanalysis-schuster-degen.pdf", plot=reanalysis_plot, width=8, height=3)


# Original model by S&D (2019):

print("Re-analysis of S&D (2019): model using AUC")
model = lm(formula= auc_diff ~ cond + test_order + first_speaker_type + confident_speaker, data = rbind(aucs.cautious, aucs.confident))
print(summary(model))
print("Re-analysis of S&D (2019): model using LLR")
model = lm(likelihood_ratio ~ first_speaker_type + condition + test_order + confident_speaker, data=d.lr)
print(summary(model))

#############################################
# Experiment  1  #
#############################################

d.adaptation = read.csv("../../../experiments/4_id_tasks/data/5_adaptation_data-merged.csv")
d.ktt = read.csv("../../../experiments/4_id_tasks/data/1_scores_ktt-anon.csv") %>% 
  dplyr::mutate(score = case_when(is.na(exclude) | exclude != 1 ~ score)) %>%
  dplyr::select(workerid, score) %>% 
  dplyr::rename (ktt.score = score) 
d.rmet = read.csv("../../../experiments/4_id_tasks/data/2_scores_rmet-anon.csv") %>%
  dplyr::mutate(score = case_when(is.na(exclude) | exclude != 1 ~ score)) %>%
  dplyr::select("workerid", "score") %>% 
  dplyr::rename (rmet.score = score)
d.crt = read.csv("../../../experiments/4_id_tasks/data/3_crt_scores-anon.csv") %>%
  dplyr::mutate(crt = case_when(is.na(exclude) | exclude != 1 ~ crt)) %>%
  dplyr::select("workerid", "crt") %>% 
  dplyr::rename(crt.score = crt)

d.art = read.csv("../../../experiments/4_id_tasks/data/4_art_scores_with_timeouts_false_pos-anon.csv") %>%
  dplyr::mutate(art.score = case_when(is.na(exclude) | exclude != 1 ~ art.score)) %>%
  dplyr::select("workerid", "art.score") 

d.ktt.all = read.csv("../../../experiments/4_id_tasks/data/1_scores_ktt-anon.csv") %>%
  dplyr::select(workerid, score)

print(paste("Number of participants before ID test exclusions:", length(unique(d.ktt.all$workerid))))


d = d.adaptation %>% 
  merge(., d.ktt) %>% 
  merge(., d.rmet) %>% 
  merge(., d.crt) %>% 
  merge(., d.art) 

print(paste("Number of participants after ID test exclusions:", length(unique(na.omit(d)$workerid))))

d = d %>% dplyr::mutate(condition = factor(condition),
                        test_order = factor(test_order),
                        first_speaker_type = factor(first_speaker_type, levels=c("confident", "cautious")))
contrasts(d$condition) = contr.sum(2)
colnames(contrasts(d$condition)) = c("cautious")
contrasts(d$test_order) = contr.sum(2)
colnames(contrasts(d$test_order)) = c("parallel")
contrasts(d$first_speaker_type) = contr.sum(2)
colnames(contrasts(d$first_speaker_type)) = c("cautious")

d$art.score = myCenter(myNorm(d$art.score))
d$ktt.score = myCenter(myNorm(d$ktt.score))
d$rmet.score = myCenter(myNorm(d$rmet.score))
d$crt.score = myCenter(myNorm(d$crt.score))

model_full = lmer(formula= likelihood_ratio ~ condition + 
               test_order + 
               first_speaker_type + 
               prior_likelihood_ratio + 
               art.score * condition +
               ktt.score * condition +
               rmet.score * condition +
               crt.score * condition +
               (1| workerid), 
             data = na.omit(d))

print(summary(model_full))

step_res = lmerTest::step(model_full)
final = lmerTest::get_model(step_res)
print(summary(final))

model = lmer(formula= likelihood_ratio ~ condition + 
               test_order + 
               first_speaker_type + 
               prior_likelihood_ratio + 
               ktt.score * condition +
               (1| workerid), 
             data = d %>% filter(!is.na(ktt.score)))

step_res = lmerTest::step(model)
final.exp1 = lmerTest::get_model(step_res)
print(summary(final.exp1))

############################################
# Experiment 2                             #
############################################

d.adaptation = read.csv("../../../experiments/5_talker_specific_adaptation_prior_exp_test_ktt/data/adaptation_data-merged.csv")
d.ktt = read.csv("../../../experiments/5_talker_specific_adaptation_prior_exp_test_ktt/data/scores_ktt-anon.csv") %>% 
  dplyr::select(workerid, score) %>% 
  dplyr::rename (ktt.score = score) %>%
  dplyr::filter(ktt.score > 19)

print(paste("Number of participants before ID test exclusions:", length(unique(d.adaptation$workerid))))

d = d.adaptation %>% 
  merge(., d.ktt) 


print(paste("Number of participants after ID test exclusions:", length(unique(d$workerid))))

d.diff = d %>% 
  dplyr::filter(condition == "cautious") %>%
  dplyr::rename(likelihood_ratio.cautious = likelihood_ratio) %>%
  dplyr::rename(most_likely_model.cautious = most_likely_model) %>%
  dplyr::select(workerid, ktt.score, likelihood_ratio.cautious, most_likely_model.cautious) %>%
  merge(d %>% 
          dplyr::filter(condition == "confident") %>%
          dplyr::rename(likelihood_ratio.confident = likelihood_ratio) %>%
          dplyr::rename(most_likely_model.confident = most_likely_model) %>% 
          dplyr::select(workerid, likelihood_ratio.confident, most_likely_model.confident)) %>%
  mutate(likelihood_diff = likelihood_ratio.confident - likelihood_ratio.cautious) %>%
  mutate(most_likely_model.cautious.correct = most_likely_model.cautious == "cautious") %>%
  mutate(most_likely_model.confident.correct = most_likely_model.confident == "confident") %>%
  mutate(adapter_type = case_when(
    most_likely_model.confident.correct & most_likely_model.cautious.correct ~ "adapted to both speakers",
    most_likely_model.confident.correct & !most_likely_model.cautious.correct ~ "always confident",
    !most_likely_model.confident.correct & most_likely_model.cautious.correct ~ "always cautious",
    TRUE ~ "adapted to wrong speakers"
  ))



corr_plot = d.diff %>% 
  mutate(adapter_type = factor(adapter_type, levels=c("adapted to wrong speakers", "always cautious", "always confident", "adapted to both speakers"))) %>%
  ggplot(aes(x=likelihood_diff, y=ktt.score)) + 
  geom_point(aes(col=adapter_type, pch=adapter_type), size=3) +
  geom_smooth(method="lm") +
  theme(legend.position = "bottom") + 
  xlab ("Difference in likelihood ratio across two test conditions\n(LLR confident - LLR cautious)") + 
  ylab("Keep track task score") +
  guides(pch=guide_legend(title="Adapter type"), col=guide_legend("Adapter type"))

ggsave(filename="exp2-ktt-LLD.pdf", plot=corr_plot, width=8, height=6)

d = d %>% dplyr::mutate(condition = factor(condition),
                        test_order = factor(test_order),
                        first_speaker_type = factor(first_speaker_type, levels=c("confident", "cautious")))
contrasts(d$condition) = contr.sum(2)
colnames(contrasts(d$condition)) = c("cautious")
contrasts(d$test_order) = contr.sum(2)
colnames(contrasts(d$test_order)) = c("parallel")
contrasts(d$first_speaker_type) = contr.sum(2)
colnames(contrasts(d$first_speaker_type)) = c("cautious")

d$ktt.score = myCenter(myNorm(d$ktt.score))



model.exp2 = lmer(formula= likelihood_ratio ~ condition + 
               first_speaker_type + 
               ktt.score * condition +
               (1| workerid), 
             data = d)


# write coefficents to tex file
texreg::texreg(c(model_full,final.exp1, model.exp2), 
               single.row=T, 
               include.aic = F, 
               include.bic = F,
               include.loglik = F, 
               include.nobs = F,
               include.groups = F, 
               include.variance = F, 
               booktabs=T, file = "model-coefficients.tex", 
               custom.model.names = c("Exp 1: Full model ($n=67$)", "Exp. 1: Reduced model ($n=96$)", "Exp. 2 ($n=91$)"),
               custom.coef.map = list("(Intercept)" = "Intercept", 
                                     "conditioncautious" = "Condition", 
                                     "test_orderparallel"= "Test order", 
                                     "first_speaker_typecautious" = "Most recent speaker", 
                                     "prior_likelihood_ratio" = "Prior likelihood ratio", 
                                     "ktt.score"= "KTT", 
                                     "art.score" = "ART", 
                                     "rmet.score" = "RMET", 
                                     "crt.score" = "CRT", 
                                     "conditioncautious:ktt.score" =   "Condition:KTT",
                                     "conditioncautious:art.score" = "Condition:ART",
                                     "conditioncautious:rmet.score" = "Condition:RMET",
                                     "conditioncautious:crt.score" = "Condition:CRT"
),
      label = "tab:model-coeff", 
      table = F,
      use.packages = F
    )


##########################
# Additional analyses
##########################

d.adaptation = read.csv("../../../experiments/5_talker_specific_adaptation_prior_exp_test_ktt/data/adaptation_data-merged.csv")
d.ktt = read.csv("../../../experiments/5_talker_specific_adaptation_prior_exp_test_ktt/data/scores_ktt-anon.csv") %>% 
  dplyr::select(workerid, score) %>% 
  dplyr::rename (ktt.score = score) 


d = d.adaptation %>% 
  merge(., d.ktt) 

d = d %>% dplyr::mutate(condition = factor(condition),
                        test_order = factor(test_order),
                        first_speaker_type = factor(first_speaker_type, levels=c("confident", "cautious")))
contrasts(d$condition) = contr.sum(2)
colnames(contrasts(d$condition)) = c("cautious")
contrasts(d$test_order) = contr.sum(2)
colnames(contrasts(d$test_order)) = c("parallel")
contrasts(d$first_speaker_type) = contr.sum(2)
colnames(contrasts(d$first_speaker_type)) = c("cautious")

d$ktt.score = myCenter(myNorm(d$ktt.score))

model.exp2a = lmer(formula= likelihood_ratio ~ condition + 
                    first_speaker_type + 
                    ktt.score * condition +
                    (1| workerid), 
                  data = d)
print(summary(model.exp2a))


######################################
# Adaptation plots                   #
######################################


#### Exp. 1
trials = remove_quotes(format_data(read.csv("../../../experiments/3_talker_specific_adaptation_prior_exp_test_id/data/3_talker_specific_adaptation_prior_exp_test_id-trials.csv")))
exp_trials = remove_quotes(read.csv("../../../experiments/3_talker_specific_adaptation_prior_exp_test_id/data/3_talker_specific_adaptation_prior_exp_test_id-exp_trials.csv"))
conditions = read.csv("../../../experiments/3_talker_specific_adaptation_prior_exp_test_id/data/3_talker_specific_adaptation_prior_exp_test_id-condition.csv")
subj_info = read.csv("../../../experiments/3_talker_specific_adaptation_prior_exp_test_id/data/3_talker_specific_adaptation_prior_exp_test_id-subject_information.csv") %>% select(workerid, noticed_manipulation)

trials = merge(trials, conditions %>% select(workerid, condition), by=c("workerid"))

trials = merge(trials, subj_info, by=c("workerid"))

trials[, c("test_order", "first_speaker_type", "second_speaker_type", "confident_speaker")] = str_split(trials$condition, "_", simplify=T)

EXCLUDE_BELOW = 11

catch_trial_perf = get_correct_catch_trial_counts(exp_trials)

exclude = catch_trial_perf %>%
  filter(catch_perf < EXCLUDE_BELOW) %>%
  .$workerid


print(paste("Excluded", length(exclude), "participants based on catch-trial performance."))


#final data
d.exp3 = trials %>% filter(., !(workerid %in% exclude))


trials = remove_quotes(format_data(read.csv("../../../experiments/1_talker_specific_adaptation_prior_exp_test/data/1_talker_specific_adaptation_prior_exp_test-trials.csv")))
exp_trials = remove_quotes(read.csv("../../../experiments/1_talker_specific_adaptation_prior_exp_test/data/1_talker_specific_adaptation_prior_exp_test-exp_trials.csv"))

conditions = read.csv("../../../experiments/1_talker_specific_adaptation_prior_exp_test/data/1_talker_specific_adaptation_prior_exp_test-condition.csv")

subj_info = read.csv("../../../experiments/1_talker_specific_adaptation_prior_exp_test/data/1_talker_specific_adaptation_prior_exp_test-subject_information.csv") %>% select(workerid, noticed_manipulation)

trials = merge(trials, conditions %>% select(workerid, condition), by=c("workerid"))

trials = merge(trials, subj_info, by=c("workerid"))

trials[, c("test_order", "first_speaker_type", "second_speaker_type", "confident_speaker")] = str_split(trials$condition, "_", simplify=T)



EXCLUDE_BELOW = 0.75

catch_trial_perf = get_correct_catch_trial_counts(exp_trials)



exclude = catch_trial_perf %>%
  filter(catch_prop < EXCLUDE_BELOW) %>%
  .$workerid


print(paste("Excluded", length(exclude), "participants based on catch-trial performance in Exp 1."))


#final data
d.exp1 = trials %>% filter(., !(workerid %in% exclude))

#combine data from exp1 and exp3

d = rbind(d.exp1, d.exp3)

included_participants = read.csv("../../../experiments/4_id_tasks/data/A_included_participants.csv")$x

d.filtered = d %>% 
  filter (workerid %in% included_participants) %>%
  filter(speaker_type != "prior") %>%
  filter(modal != "other")



d_means = d.filtered %>%
  group_by(workerid, percentage_blue, modal, speaker_type) %>% 
  summarise(participant_mean = mean(rating)) %>%
  group_by(percentage_blue, modal, speaker_type) %>%
  summarise(mu = mean(participant_mean),
            ci_high = ci.high(participant_mean), 
            ci_low = ci.low(participant_mean))


p1 = ggplot(d_means, aes(x=percentage_blue, y=mu, col=modal, lty=speaker_type)) + 
  xlab("% blue gumballs") +
  ylab("mean ratings") +
  geom_errorbar(aes(ymin=mu-ci_low, ymax=mu+ci_high), width=3, lty=1) +
  geom_line() +
  geom_point(size=1) +
  guides(col=guide_legend(title="Expr."), lty=guide_legend("Speaker")) +
  scale_color_manual(values=c("#7CB637",  "#4381C1"))  +
  theme(legend.position="bottom") + 
  geom_vline(xintercept=60, lty=3, col="gray", size=1)

plot(p1)

ggsave(filename="exp1-adaptation-plot.pdf", width=7, height=5, plot=p1)

#### Exp. 2

# Load data

trials = remove_quotes(format_data(read.csv("../../../experiments/5_talker_specific_adaptation_prior_exp_test_ktt/data/5_talker_specific_adaptation_prior_exp_test_ktt-trials.csv")))
exp_trials = remove_quotes(read.csv("../../../experiments/5_talker_specific_adaptation_prior_exp_test_ktt/data/5_talker_specific_adaptation_prior_exp_test_ktt-exp_trials.csv"))
conditions = read.csv("../../../experiments/5_talker_specific_adaptation_prior_exp_test_ktt/data/5_talker_specific_adaptation_prior_exp_test_ktt-condition.csv")


trials = merge(trials, conditions %>% select(workerid, condition), by=c("workerid"))


trials[, c("test_order", "first_speaker_type", "second_speaker_type", "confident_speaker")] = str_split(trials$condition, "_", simplify=T)
EXCLUDE_BELOW = 11

catch_trial_perf = get_correct_catch_trial_counts(exp_trials)

exclude = catch_trial_perf %>%
  filter(catch_perf < EXCLUDE_BELOW) %>%
  .$workerid

exclude = cbind(exclude, c(1640)) # some technical glitch with that participant
d = trials %>% filter(., !(workerid %in% exclude))


included_participants = read.csv("../../../experiments/5_talker_specific_adaptation_prior_exp_test_ktt/data/A_included_participants.csv")$x

d.filtered = d %>% 
  filter (workerid %in% included_participants) %>%
  filter(speaker_type != "prior") %>%
  filter(modal != "other")



d_means = d.filtered %>%
  group_by(workerid, percentage_blue, modal, speaker_type) %>% 
  summarise(participant_mean = mean(rating)) %>%
  group_by(percentage_blue, modal, speaker_type) %>%
  summarise(mu = mean(participant_mean),
            ci_high = ci.high(participant_mean), 
            ci_low = ci.low(participant_mean))


p1 = ggplot(d_means, aes(x=percentage_blue, y=mu, col=modal, lty=speaker_type)) + 
  xlab("% blue gumballs") +
  ylab("mean ratings") +
  geom_errorbar(aes(ymin=mu-ci_low, ymax=mu+ci_high), width=3, lty=1) +
  geom_line() +
  geom_point(size=1) +
  guides(col=guide_legend(title="Expr."), lty=guide_legend("Speaker")) +
  scale_color_manual(values=c("#7CB637",  "#4381C1"))  +
  theme(legend.position="bottom") + 
  geom_vline(xintercept=60, lty=3, col="gray", size=1)

plot(p1)

ggsave(filename="exp2-adaptation-plot.pdf", width=7, height=5, plot=p1)


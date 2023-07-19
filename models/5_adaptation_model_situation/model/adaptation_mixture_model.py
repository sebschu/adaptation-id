import numpy as np
from scipy.stats import beta, uniform, bernoulli, norm, truncnorm, expon
import time
import json
import sys
import os
import argparse
import copy

from threshold_model import ThresholdModel

SPEAKERS = ["speaker_1", "speaker_2"]
PARAM_SETS = SPEAKERS + ["situation"]


class AdaptationModel(ThresholdModel):
    def __init__(self, config, output_path, run):
        super().__init__(config, output_path, run)

    def compute_likelihood(self, params, speaker=None):
        log_lkhood = 0
        
        n_utt = 3 if "expr_idx1" in params["situation"] else len(self.expressions) + 1
        speaker_probs = np.zeros(
            (2, n_utt, self.probabilities_len), dtype=np.float64)
        for key in params:
            partial_speaker_probs = np.exp(
                self.log_speaker_dist(**params[key]))
            if key in SPEAKERS and speaker in (None, key):
                speaker_probs[SPEAKERS.index(
                    key)] += partial_speaker_probs * self.config["attention_weight"]
            elif key == "situation":
                speaker_probs += partial_speaker_probs * \
                    (1 - self.config["attention_weight"])
        speaker_probs = np.log(speaker_probs)


        n_utt = min(n_utt, len(self.expressions))

        for s, data in self.data.items():
            speaker_idx = SPEAKERS.index(s)
            log_lkhood += np.sum(np.multiply(data,
                                 speaker_probs[speaker_idx, 0:n_utt, :]))

        return log_lkhood

    def load_data(self):
        data = {}
        for speaker in self.config["exposure_trials"]:
            data[speaker] = np.zeros(
                (len(self.expressions), self.probabilities_len), dtype=np.float64)
        # exposure trials are of the form ["utterance", prob, count]
            for t in self.config["exposure_trials"][speaker]:
                data[speaker][self.expressions2idx[t[0]], self.prob2idx[t[1]]] += t[2]
        return data

    def shape_a(self, mu, nu):
        return mu * nu

    def shape_b(self, mu, nu):
        return (1-mu) * nu

    def beta_mu(self, a, b):
        return a / (a+b)

    def beta_var(self, a, b):
        return a * b / ((a + b)**2 * (a + b + 1))

    def beta_nu(self, a, b):
        return a + b
    
    def beta_mu_nu(self, a, b):
        return self.beta_mu(a, b), self.beta_nu(a, b)

    def theta_mu_param_prior(self, old_val):
        width = self.config["proposal_widths"]["theta_mu"]
        if width == 0:
            return old_val
        a = (0 - old_val) / width
        b = (1 - old_val) / width
        return truncnorm(a, b, loc=old_val, scale=width).rvs()

    def theta_nu_param_prior(self, old_val, min_val):
        width = self.config["proposal_widths"]["theta_nu"]
        if width == 0:
            return old_val
        old_val = np.log(old_val)
        a = (np.log(min_val) - old_val) / width
        b = (100 - old_val) / width
        return np.exp(truncnorm(a, b, loc=old_val, scale=width).rvs())

    def rat_alpha_prior(self, old_val):
        width = self.config["proposal_widths"]["rat_alpha"]
        if width == 0:
            return old_val
        old_val = np.log(old_val)
        return np.exp(norm(old_val, width).rvs())

    def utt_other_prob_prior(self, old_val):
        if not self.config["utt_other_prob_estimate"]:
            return old_val
        width = self.config["proposal_widths"]["utt_other_prob"]
        old_val = np.log(old_val)
        return np.exp(norm(old_val, width).rvs())

    def init_trace(self):
        theta_mus = uniform(0, 1).rvs(len(self.expressions))
        theta_nus = uniform(1, 5).rvs(len(self.expressions))
        theta_alphas = np.ones(theta_mus.shape)
        theta_betas = np.ones(theta_mus.shape)
        costs = uniform(0, 7).rvs(len(self.expressions))
        rat_alpha = self.config["rat_alpha_init"]
        utt_other_prob = self.config["utt_other_prob_init"]
        noise_strength = self.config["noise_strength_init"]

        for i, utt in enumerate(self.config["utterances"]):
            if not utt["has_theta"]:
                theta_mus[i] = self.beta_mu(1, 1)
                theta_nus[i] = self.beta_nu(1, 1)
            else:
                if utt["mu_init"] != "random":
                    theta_mus[i] = utt["mu_init"]
                if utt["nu_init"] != "random":
                    theta_nus[i] = utt["nu_init"]

            theta_alphas[i] = self.shape_a(theta_mus[i], theta_nus[i])
            theta_betas[i] = self.shape_b(theta_mus[i], theta_nus[i])

            if not utt["has_cost"]:
                costs[i] = 0
            else:
                if utt["cost_init"] != "random":
                    costs[i] = utt["cost_init"]

        params = {
            "utt_costs": costs,
            "rat_alpha": rat_alpha,
            "theta_alphas": theta_alphas,
            "theta_betas": theta_betas,
            "utt_other_prob": utt_other_prob,
            "noise_strength": noise_strength
        }

        return params

    def compute_trans_probs(self, src_params, tgt_params):
        log_prob = 0
        theta_mu_width = self.config["proposal_widths"]["theta_mu"]
        theta_nu_width = self.config["proposal_widths"]["theta_nu"]
        cost_width = self.config["proposal_widths"]["cost"]
        rat_alpha_width = self.config["proposal_widths"]["rat_alpha"]
        utt_other_width = self.config["proposal_widths"]["utt_other_prob"]
        for i, utt in enumerate(self.config["utterances"]):
            if utt["has_theta"] and utt["prior"]["mu_sd"] > 0 and theta_mu_width > 0:
                src_mu, src_nu = self.beta_mu_nu(src_params["theta_alphas"][i], src_params["theta_betas"][i])
                tgt_mu, tgt_nu = self.beta_mu_nu(tgt_params["theta_alphas"][i], tgt_params["theta_betas"][i])
                a = (0 - src_mu) / theta_mu_width
                b = (1 - src_mu) / theta_mu_width
                log_prob += truncnorm(a, b, loc=src_mu,
                                      scale=theta_mu_width).logpdf(tgt_mu)

                src = np.log(src_nu)
                tgt = np.log(tgt_nu)
                a = (np.log(utt["nu_init"]) - src) / theta_nu_width
                b = (100 - src) / theta_nu_width
                log_prob += truncnorm(a, b, loc=src,
                                      scale=theta_nu_width).logpdf(tgt)
            if utt["has_cost"] and "copy_cost" not in utt and utt["prior"]["cost_sd"] > 0 and cost_width > 0:
                src = np.log(src_params["utt_costs"][i])
                tgt = np.log(tgt_params["utt_costs"][i])
                log_prob += norm(src, cost_width).logpdf(tgt)

        if (self.config["rat_alpha_estimate"]):
            src = np.log(src_params["rat_alpha"])
            tgt = np.log(tgt_params["rat_alpha"])
            log_prob += norm(src, rat_alpha_width).logpdf(tgt)

        if self.config["utt_other_prob_estimate"]:
            src = np.log(src_params["utt_other_prob"])
            tgt = np.log(src_params["tgt_params"])
            log_prob += norm(src, utt_other_width).logpdf(tgt)

        return log_prob

    def compute_prior(self, params):

        log_prior = 0
        for param_key in params.keys():
            for i, utt in enumerate(self.config["utterances"]):
                if utt["has_theta"] and utt["prior"]["mu_sd"] > 0:
                    theta_mu = self.beta_mu(
                        params[param_key]["theta_alphas"][i], params[param_key]["theta_betas"][i])
                    theta_nu = self.beta_nu(
                        params[param_key]["theta_alphas"][i], params[param_key]["theta_betas"][i])
                    log_prior += norm(utt["prior"]["mu_mu"],
                                      utt["prior"]["mu_sd"]).logpdf(theta_mu)
                    log_prior += expon(loc=utt["nu_init"]-0.00000001,
                                       scale=self.config["theta_nu_scale"]).logpdf(theta_nu)

                if utt["has_cost"] and "copy_cost" not in utt and utt["prior"]["cost_sd"] > 0:
                    log_prior += norm(np.log(utt["prior"]["cost_mu"]), utt["prior"]["cost_sd"]).logpdf(
                        np.log(params[param_key]["utt_costs"][i]))

            if (self.config["rat_alpha_estimate"]):
                log_prior += norm(self.config["rat_alpha_prior"]["rat_alpha_mu"],
                                  self.config["rat_alpha_prior"]["rat_alpha_sd"]).logpdf(params[param_key]["rat_alpha"])

            if self.config["utt_other_prob_estimate"]:
                log_prior += norm(np.log(self.config["utt_other_prob_prior"]["utt_other_prob_mu"]),
                                  self.config["utt_other_prob_prior"]["utt_other_prob_sd"]).logpdf(np.log(params[param_key]["utt_other_prob"]))

        return log_prior

    def make_sample(self, params):
        full_sample = {}
        for param_key, param_val in params.items():
            sample = {
                "rat_alpha": param_val["rat_alpha"],
                "utt_other_prob": param_val["utt_other_prob"],
                "noise_strength": param_val["noise_strength"]
            }
            for i, utt in enumerate(self.config["utterances"]):
                sample["alpha_" + utt["form"]] = param_val["theta_alphas"][i]
                sample["beta_" + utt["form"]] = param_val["theta_betas"][i]
                sample["cost_" + utt["form"]] = param_val["utt_costs"][i]

            full_sample[param_key] = sample
        return full_sample

    def run_mcmc(self):
        acceptance = 0
        samples = []

        old_params = {}
        new_params = {}
        for param_key in PARAM_SETS:
            old_params[param_key] = self.init_trace()
            new_params[param_key] = copy.deepcopy(old_params[param_key])

        prior = self.compute_prior(old_params)
        old_likelihood = self.compute_likelihood(old_params) + prior
        sample = self.make_sample(old_params)

        iterations = self.config["iterations"]
        burn_in = self.config["burn_in"] if "burn_in" in self.config else 0

        log_file_name = os.path.join(
            self.output_path, "run{}.log".format(self.run))
        output_file_name = os.path.join(
            self.output_path, "run{}_output.json".format(self.run))

        log_file = open(log_file_name, "w")

        n_param_sets = len(PARAM_SETS)
        
        for it in range(iterations):
            param_key = PARAM_SETS[it % n_param_sets]
            if it > 0 and it % 100 == 0:
                print("Iteration: {} ".format(it), file=log_file)
                print("Acceptance rate: {}".format(
                    acceptance * 1.0 / it), file=log_file)
                log_file.flush()

            for param_name in old_params[param_key]:
                new_params[param_key][param_name] = copy.copy(old_params[param_key][param_name])
                
            for i, utt in enumerate(self.config["utterances"]):
                if utt["has_theta"] and utt["prior"]["mu_sd"] > 0:
                    old_theta_mu, old_theta_nu = self.beta_mu_nu(old_params[param_key]["theta_alphas"][i], old_params[param_key]["theta_betas"][i])
                    new_theta_mu = self.theta_mu_param_prior(old_theta_mu)
                    new_theta_nu = self.theta_nu_param_prior(
                        old_theta_nu, utt["nu_init"])
                    new_params[param_key]["theta_alphas"][i] = self.shape_a(new_theta_mu, new_theta_nu)
                    new_params[param_key]["theta_betas"][i] = self.shape_b(new_theta_mu, new_theta_nu)
                if utt["has_cost"] and "copy_cost" not in utt and utt["prior"]["cost_sd"] > 0:
                    new_params[param_key]["utt_costs"][i] = self.cost_prior(old_params[param_key]["utt_costs"][i])
            for i, utt in enumerate(self.config["utterances"]):
                if utt["has_cost"] and "copy_cost" in utt:
                    new_params[param_key]["utt_costs"][i] = new_params[param_key]["utt_costs"][self.expressions2idx[utt["copy_cost"]]]

            if self.config["rat_alpha_estimate"]:
                new_params[param_key]["rat_alpha"] = self.rat_alpha_prior(old_params[param_key]["rat_alpha"])
            else:
                new_params[param_key]["rat_alpha"] = old_params[param_key]["rat_alpha"]

            if self.config["utt_other_prob_estimate"]:
                new_params[param_key]["utt_other_prob"] = self.utt_other_prob_prior(old_params[param_key]["utt_other_prob"])
            else:
                new_params[param_key]["utt_other_prob"] = old_params[param_key]["utt_other_prob"]

            prior = self.compute_prior(new_params)
            new_likelihood = self.compute_likelihood(new_params) + prior
            accept = new_likelihood > old_likelihood
            if not accept:
                bwd_prob = self.compute_trans_probs(new_params[param_key],
                                                    old_params[param_key])
                fwd_prob = self.compute_trans_probs(old_params[param_key],
                                                    new_params[param_key])
                likelihood_ratio = new_likelihood - old_likelihood - fwd_prob + bwd_prob
                u = np.log(uniform(0, 1).rvs())
                if u < likelihood_ratio:
                    accept = True
                    
            if accept:
                old_likelihood = new_likelihood
                sample = self.make_sample(new_params)
                acceptance += 1
                old_params[param_key] = copy.copy(new_params[param_key])


            if it > burn_in and it % 10 == 0:
                samples.append(sample)
                if len(samples) % 1000 == 0:
                    json.dump(samples, open(output_file_name, "w"))

            if it % 20000 == 0:
                self.speaker_matrix_cache.clear()
                self.theta_prior_cache.clear()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()

    out_dir = args.out_dir
    config_file_path = os.path.join(out_dir, "config.json")
    config = json.load(open(config_file_path, "r"))

    model = AdaptationModel(config, out_dir, args.run)
    model.run_mcmc()


if __name__ == '__main__':
    main()

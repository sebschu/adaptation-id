import numpy as np
from scipy.stats import beta, uniform, bernoulli, norm
import time
import json
import sys
import os
import argparse
import copy
import csv
import glob
import re

from adaptation_mixture_model import AdaptationModel


class ModelLikelihood(AdaptationModel):
    def __init__(self, config, output_path, run, filenames=None):
        super().__init__(config, output_path, run)
        self.filenames = filenames
        self.load_mcmc_samples()

    def load_mcmc_samples(self):
        self.mcmc_samples = []
        if self.filenames is None:
            samples_file_path = os.path.join(self.output_path, "samples.json")
            self.mcmc_samples = json.load(open(samples_file_path, "r"))
        else:
            for samples_file_path in glob.glob(os.path.join(self.output_path, self.filenames)):
                self.mcmc_samples.extend(
                    json.load(open(samples_file_path, "r")))

    # Data loading and pre-processing

    def load_data(self, speaker=None):
        if not os.path.exists(self.config["data_path"]):
            print(
                f"WARNING: Data file at {self.config['data_path']} does not exist!")
            return None
        raw_data = json.load(open(self.config["data_path"], "r"))
        count_arrays = dict()
   
        for d in raw_data["obs"]:
            if speaker not in count_arrays:
                count_arrays[speaker] = np.zeros((3, self.probabilities_len))

            col_idx = self.prob2idx[d["percentage_blue"]]
            row_idx = 2 if d["modal"] == "other" else 0 if d["modal"] == d["modal1"] else 1
            count_arrays[speaker][row_idx, col_idx] += 1
        return count_arrays

    def get_params(self, sample):
        params = {}
        
        pair = self.config["conditions"][0].split("-")
        expr_idx1 = self.expressions2idx[pair[0]]
        expr_idx2 = self.expressions2idx[pair[1]]

        
        for param_key, c_sample in sample.items():
            params[param_key] = {
                "rat_alpha": c_sample["rat_alpha"],
                "utt_other_prob": c_sample["utt_other_prob"],
                "noise_strength": c_sample["noise_strength"]
            }

            costs = []
            theta_alphas = []
            theta_betas = []
            for i, utt in enumerate(self.config["utterances"]):
                theta_alphas.append(c_sample["alpha_" + utt["form"]])
                theta_betas.append(c_sample["beta_" + utt["form"]])
                costs.append(c_sample["cost_" + utt["form"]])
            params[param_key]["utt_costs"] = costs
            params[param_key]["theta_alphas"] = theta_alphas
            params[param_key]["theta_betas"] = theta_betas
            
            params[param_key]["expr_idx1"] = expr_idx1
            params[param_key]["expr_idx2"] = expr_idx2

        return params

    def compute_likelihood_for_samples(self, workerid, condition, out_filename):

        speaker = "speaker_1" if condition == "cautious" else "speaker_2"

        likelihood = 0
        params = self.get_params(self.mcmc_samples[0])
        first_likelihood = self.compute_likelihood(params, speaker=speaker)

        for it, sample in enumerate(self.mcmc_samples):
            params = self.get_params(sample)
            if it > 0:
                likelihood = np.logaddexp(self.compute_likelihood(
                    params, speaker=speaker) - first_likelihood, likelihood)
            else:
                likelihood = self.compute_likelihood(
                    params, speaker=speaker) - first_likelihood

            # print(likelihood)

            if it > 0 and it % 100 == 0:
                print("Iteration: ", it)

            if it > 0 and it % 20000 == 0:
                self.speaker_matrix_cache.clear()
                self.theta_prior_cache.clear()
        likelihood += first_likelihood - np.log(len(self.mcmc_samples))
        with open(out_filename, "a") as out_f:
            print(f"{workerid},{condition},{likelihood}", file=out_f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir_root", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_filename", required=False)
    parser.add_argument("--global_priors", required=False, action="store_true")

    args = parser.parse_args()

    model = None

    # out_filename = args.out_filename if args.out_filename is not None else os.path.join(args.out_dir, "likelihood")
    # with open(out_filename, "w") as out_f:
    #  print("workerid,condition,likelihood", file=out_f)

    if not args.global_priors:
        model_path = os.path.join(
            args.out_dir_root, "") + "subject*/*output.json"
    else:
        model_path = os.path.join(args.out_dir_root, "") + "*output.json"
    for f in glob.glob(model_path):
        print(f"Processing {f}...")
        if not args.global_priors:
            workerid = re.findall("[1-9][0-9][0-9][0-9]", f)[0]
        out_dir = os.path.dirname(f)
        config_file_path = os.path.join(os.path.dirname(f), "config.json")
        config = json.load(open(config_file_path, "r"))
        model = ModelLikelihood(config, out_dir, "", filenames="*output.json")
        out_filename = os.path.join(out_dir, "likelihood")
        for condition in ["cautious", "confident"]:
          speaker = "speaker_1" if condition == "cautious" else "speaker_2"
          if not args.global_priors:
              data_path = os.path.join(
                  args.data_dir, f"indiv_differences_adaptation_{condition}-worker-{workerid}.json")
              config["data_path"] = data_path
              model.config = config
              model.data = model.load_data(speaker=speaker)
              model.compute_likelihood_for_samples(
                  workerid, condition, out_filename)
          else:
              if condition == "cautious":
                with open(out_filename, "w") as out_f:
                    print("workerid,condition,likelihood", file=out_f)
              data_paths = os.path.join(
                  args.data_dir, f"indiv_differences_adaptation_{condition}-worker-*.json")
              for p in glob.glob(data_paths):
                  workerid = re.findall("[1-9][0-9][0-9][0-9]", p)[0]
                  config["data_path"] = p
                  model.config = config
                  model.data = model.load_data(speaker=speaker)
                  model.compute_likelihood_for_samples(
                      workerid, condition, out_filename)


if __name__ == '__main__':
    main()

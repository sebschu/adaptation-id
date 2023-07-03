import numpy as np
from scipy.stats import beta, uniform, bernoulli, norm
import time, json
import sys
import os
import argparse
import copy
import csv
import glob
import re

from threshold_model import ThresholdModel

class ModelLikelihood(ThresholdModel):
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
         self.mcmc_samples.extend(json.load(open(samples_file_path, "r")))
      
  
  def get_params(self, sample):
      rat_alpha = sample["rat_alpha"]
      utt_other_prob = sample["utt_other_prob"]
      noise_strength = sample["noise_strength"]
      
      costs = []
      theta_alphas = []
      theta_betas = []
      for i, utt in enumerate(self.config["utterances"]):
        theta_alphas.append(sample["alpha_" + utt["form"]])
        theta_betas.append(sample["beta_" + utt["form"]]) 
        costs.append(sample["cost_" + utt["form"]])
      return (theta_alphas, theta_betas, costs, rat_alpha, utt_other_prob, noise_strength)
  
  def compute_likelihood_for_samples(self, workerid, condition, out_filename):
    
    likelihood = 0
    theta_alphas, theta_betas, costs, rat_alpha, utt_other_prob, noise_strength = self.get_params(self.mcmc_samples[0])
    first_likelihood = self.compute_likelihood(costs, rat_alpha, theta_alphas, theta_betas, utt_other_prob, noise_strength)
    
    for it, sample in enumerate(self.mcmc_samples):
      theta_alphas, theta_betas, costs, rat_alpha, utt_other_prob, noise_strength = self.get_params(sample)
      if it > 0:
        likelihood = np.logaddexp(self.compute_likelihood(costs, rat_alpha, theta_alphas, theta_betas, utt_other_prob, noise_strength) - first_likelihood, likelihood)
      else:
        likelihood = self.compute_likelihood(costs, rat_alpha, theta_alphas, theta_betas, utt_other_prob, noise_strength) - first_likelihood
        
      #print(likelihood)
      
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

  #out_filename = args.out_filename if args.out_filename is not None else os.path.join(args.out_dir, "likelihood")
  #with open(out_filename, "w") as out_f:
  #  print("workerid,condition,likelihood", file=out_f)
  
  if not args.global_priors:
    model_path = os.path.join(args.out_dir_root, "") + "subject*/*/*output.json"
  else:
    model_path = os.path.join(args.out_dir_root, "") + "*/*/*output.json"
  for f in glob.glob(model_path):
    print(f"Processing {f}...")
    if not args.global_priors:
      workerid = re.findall("[1-9][0-9][0-9][0-9]", f)[0]
    condition = re.findall("(cautious|confident)", f)[0] 
    out_dir = os.path.dirname(f)
    config_file_path = os.path.join(os.path.dirname(f), "config.json")
    config = json.load(open(config_file_path, "r"))
    model = ModelLikelihood(config, out_dir, "", filenames="*output.json")
    out_filename = os.path.join(out_dir, "likelihood")
    if not args.global_priors:
      data_path = os.path.join(args.data_dir, f"indiv_differences_adaptation_{condition}-worker-{workerid}.json")
      config["data_path"] = data_path
      model.config = config
      model.data = model.load_data()
      model.compute_likelihood_for_samples(workerid, condition, out_filename)
    else:
      with open(out_filename, "w") as out_f:
        print("workerid,condition,likelihood", file=out_f)
      data_paths = os.path.join(args.data_dir, f"indiv_differences_adaptation_{condition}-worker-*.json")
      for p in glob.glob(data_paths):
        workerid = re.findall("[1-9][0-9][0-9][0-9]", p)[0] 
        config["data_path"] = p
        model.config = config
        model.data = model.load_data()
        model.compute_likelihood_for_samples(workerid, condition, out_filename)
        
         


    
        
if __name__ == '__main__':
  main()
  

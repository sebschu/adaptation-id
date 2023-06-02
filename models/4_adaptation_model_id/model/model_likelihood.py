import numpy as np
from scipy.stats import beta, uniform, bernoulli, norm
import time, json
import sys
import os
import argparse
import copy
import csv
import glob

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
  parser.add_argument("--out_dir", required=True)
  parser.add_argument("--data_dir", required=True)
  parser.add_argument("--filenames", required=False)
  parser.add_argument("--out_filename", required=False)
  args = parser.parse_args()
  
  out_dir = args.out_dir
  config_file_path = os.path.join(out_dir, "config.json")
  config = json.load(open(config_file_path, "r"))
  
  model = None

  out_filename = args.out_filename if args.out_filename is not None else os.path.join(args.out_dir, "likelihood")
  with open(out_filename, "w") as out_f:
    print("workerid,condition,likelihood", file=out_f)
  
  
  data_path = os.path.join(args.data_dir, "") + "indiv_differences_adaptation_*-worker-*.json"
  for f in glob.glob(data_path):
    print(f"Processing {f}...")
    config["data_path"] = f
    parts = f.split("-worker-")
    workerid = parts[1].replace(".json", "")
    condition = parts[0].replace(os.path.join(args.data_dir, "") + "indiv_differences_adaptation_", "")
    
    if model is None:
      model = ModelLikelihood(config, out_dir, "", filenames=args.filenames)
    
    model.config = config
    model.data = model.load_data()
    model.compute_likelihood_for_samples(workerid, condition, out_filename)  

        
if __name__ == '__main__':
  main()
  
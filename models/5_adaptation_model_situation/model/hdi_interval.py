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

class HDISampler(ThresholdModel):
  def __init__(self, config, output_path, hdi_output_file, run, filenames=None, subjectid = None):
    super().__init__(config, output_path, run)
    self.hdi_output_file = hdi_output_file
    self.filenames = filenames
    self.subjectid = subjectid
    self.load_mcmc_samples()
  
    
  def load_data(self):
    data = {"conditions": self.config["conditions"]}
    return data
  
  def load_mcmc_samples(self):
    self.mcmc_samples = []
    if self.filenames is None:
      samples_file_path = os.path.join(self.output_path, "samples.json")
      self.mcmc_samples = json.load(open(samples_file_path, "r"))
    else:
      for samples_file_path in glob.glob(os.path.join(self.output_path, self.filenames)):
         self.mcmc_samples.extend(json.load(open(samples_file_path, "r")))
      
      
    n_samples = self.config["hdi_estimation_samples"]
    self.mcmc_samples = np.random.choice(self.mcmc_samples, size=n_samples, replace=False)

  
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
  
  def generate_hdi_samples(self):
    fieldnames = ['modal', 'percentage_blue', 'cond', 'rating_pred', 'run', 'workerid']
    
    with open(self.hdi_output_file, "a") as out_f:
      writer = csv.DictWriter(out_f, fieldnames=fieldnames)
    
      for it, sample in enumerate(self.mcmc_samples):
        theta_alphas, theta_betas, costs, rat_alpha, utt_other_prob, noise_strength = self.get_params(sample)
        for cond in self.data["conditions"]:
          expr_1, expr_2 = cond.split("-")
          speaker_probs =  np.exp(self.log_speaker_dist(costs, self.expressions2idx[expr_1], self.expressions2idx[expr_2], rat_alpha, theta_alphas, theta_betas, utt_other_prob, noise_strength))
          for i in range(self.probabilities_len):
            for j, utt in enumerate([expr_1, expr_2, "other"]):
              hdi_sample = {"modal": utt, "percentage_blue": self.probabilities[i], "cond": cond, "rating_pred": speaker_probs[j, i], "run": it}
              if self.subjectid is not None:
                hdi_sample["workerid"] = self.subjectid
              writer.writerow(hdi_sample)
        
        if it > 0 and it % 100 == 0:
          print("Iteration: ", it)
          
        
        if it > 0 and it % 20000 == 0:
          self.speaker_matrix_cache.clear()
          self.theta_prior_cache.clear()
        


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--out_dir_root", required=True)
  args = parser.parse_args()
  
  out_root_dir = args.out_dir_root
  
  for condition in ["cautious", "confident"]:
  
    output_file_name = os.path.join(out_root_dir, f"hdi_samples_{condition}.csv")
    with open(output_file_name, "w") as out_f:
      fieldnames = ['modal', 'percentage_blue', 'cond', 'rating_pred', 'run', 'workerid']
      writer = csv.DictWriter(out_f, fieldnames=fieldnames)
      writer.writeheader()

    
    
    for subject_output_path in glob.glob(os.path.join(out_root_dir, f"subject**/{condition}/")):
      config_file_path = os.path.join(subject_output_path, "config.json")
      config = json.load(open(config_file_path, "r"))

      workerid = re.findall("subject[1-9][0-9][0-9][0-9]", subject_output_path)[0].replace("subject", "")
      model = HDISampler(config, subject_output_path, output_file_name, "", filenames="run1_output.json", subjectid=workerid)
      model.generate_hdi_samples()

if __name__ == '__main__':
  main()
  

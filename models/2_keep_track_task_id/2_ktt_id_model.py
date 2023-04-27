import numpy as np
from scipy.stats import norm, uniform
import random
from collections import Counter
import csv, json
import sys
import argparse
import os
import copy


class KTTIDModel(object):

    CATEGORIES = {
        "animals": ["bear", "dog", "horse", "lion", "pig", "wolf"],
        "colors": ["blue", "green", "orange", "purple", "red", "yellow"],
        "countries": ["egypt", "brazil", "china", "france", "india", "japan"],
        "distances": ["meter", "mile", "inch", "kilometer", "foot", "yard"],
        "metals": ["tin", "nickel", "steel", "lead", "zinc", "iron"],
        "relatives": ["mother", "father", "brother", "sister", "aunt", "uncle"]
        }
    
    def __init__(self, out_path, run, config_path):
        self.out_path = out_path
        self.run = run
        self.load_config(config_path)
        self.data_path = self.config["data_path"]

        self.load_data()
    
    def load_config(self, config_path):
        with open(config_path, "r", encoding="UTF-8") as f:
            self.config = json.load(f)
    
    def load_data(self):
        self.data = {}
        self.workerids = []
        with open(self.data_path, "r") as f:
            reader =  csv.DictReader(f)
            for line in reader:
                d = json.loads(json.loads(line["answer"]))
                wid = line["workerid"]
                if wid not in self.data:
                    self.data[wid] = []
                    self.workerids.append(wid)
                self.data[wid].append(d)


        

    def update(self, weights, category, item, magnitude):
        item_idx = self.CATEGORIES[category].index(item)
        weights[category][item_idx] += magnitude

    def decay(self, weights, decay_rate):
        for cat in self.CATEGORIES:
            weights[cat] = ((weights[cat] - 2) * decay_rate) + 2 

    def run_experiment(self, data, N, attention_weight, decay_rate):
        weights = {}
        for cat in self.CATEGORIES:
            n = len(self.CATEGORIES[cat])
            weights[cat] = np.ones(n) * 2.0
            
        
        within_category_update = N * attention_weight / len(data["selectedCategories"])
        other_category_update = (N - within_category_update) / len(self.CATEGORIES)

        for item in data["items"]:
            update_size = within_category_update if item["category"] in data["selectedCategories"] else other_category_update
            
            self.decay(weights, decay_rate)
            
            cat = item["category"].lower()
            item_name = item["item"].lower()
            
            self.update(weights, cat, item_name, magnitude=update_size)
            #plot_all_cats()
        
        return weights

    def init_trace(self):
        
        if not self.config["parameters"]["decay_rate"]["update"]:
            decay_rate = self.config["parameters"]["decay_rate"]["init"]
        else:
            decay_rate_mu = self.config["parameters"]["decay_rate"]["prior_mu"] #0.71
            decay_rate_sd = self.config["parameters"]["decay_rate"]["prior_sd"] #0.1
            decay_rate = random.normalvariate(decay_rate_mu, decay_rate_sd)
            decay_rate = max(0.5, min(1.0, decay_rate))
        
        if not self.config["parameters"]["N"]["update"]:
            N = self.config["parameters"]["N"]["init"]
        else:
            N_mu = self.config["parameters"]["N"]["prior_mu"] #1000
            N_sd = self.config["parameters"]["N"]["prior_sd"] #250
            N = random.normalvariate(N_mu, N_sd)
            N = max(1.0, N)
        
       
        attention_weight_mu = self.config["parameters"]["attention_weight"]["prior_mu"] #0.77
        attention_weight_sd = self.config["parameters"]["attention_weight"]["prior_sd"] #0.14
        
        attention_weights = {}
        for wid in self.workerids:
            attention_weights[wid] = max(0.5, min(1, random.normalvariate(attention_weight_mu, attention_weight_sd)))
        
        
        return (N, attention_weights, decay_rate)

    def proposal_N(self, old_val):
        if not self.config["parameters"]["N"]["update"]:
            return old_val

        width = self.config["parameters"]["N"]["proposal_width"]
        old_val = np.log(old_val)
        return np.exp(norm(old_val, width).rvs())
        

    def proposal_attention_weight(self, old_val):
        if not self.config["parameters"]["attention_weight"]["update"]:
            return old_val
        
        width = self.config["parameters"]["attention_weight"]["proposal_width"]
        a = max(.5, old_val - width)
        b = min(1, old_val + width)
        b = max(0, b - a)
        return uniform(a, b).rvs()

    
    def proposal_decay_rate(self, old_val):
        if not self.config["parameters"]["decay_rate"]["update"]:
            return old_val
        
        width = self.config["parameters"]["decay_rate"]["proposal_width"]
        a = max(.5, old_val - width)
        b = min(1, old_val + width)
        b = max(0, b - a)
        return uniform(a, b).rvs()
    

    def prior(self, N, attention_weights, decay_rate, workerid=None):
        log_prior = 0
         
        # prior over N
        if self.config["parameters"]["N"]["update"]:
            N_mu = self.config["parameters"]["N"]["prior_mu"] 
            N_sd = self.config["parameters"]["N"]["prior_sd"] 
            log_prior += norm(N_mu, N_sd).logpdf(N)
        
        #prior over decay rate
        if self.config["parameters"]["decay_rate"]["update"]:
            decay_rate_mu = self.config["parameters"]["decay_rate"]["prior_mu"] #0.71
            decay_rate_sd = self.config["parameters"]["decay_rate"]["prior_sd"] #0.01
            log_prior += norm(decay_rate_mu, decay_rate_sd).logpdf(decay_rate)
        
        attention_weight_mu = self.config["parameters"]["attention_weight"]["prior_mu"]
        attention_weight_sd = self.config["parameters"]["attention_weight"]["prior_sd"]

        if workerid is not None:
            log_prior += norm(attention_weight_mu, attention_weight_sd).logpdf(attention_weights[workerid])
        
        else:
            for wid in self.workerids:
                log_prior += norm(attention_weight_mu, attention_weight_sd).logpdf(attention_weights[wid])
        
        return log_prior


    def compute_mode(self, weights, cat):
        alphas, labels = weights[cat], self.CATEGORIES[cat]
        
        A = np.sum(alphas)
        xs = (alphas - 1) / (A - len(alphas))
        return dict(zip(labels, xs))

                

    def compute_likelihood_for_worker(self, N, attention_weights, decay_rate, workerid):
        likelihood = 0
        for d in self.data[workerid]:
                final_weights = self.run_experiment(d, N, attention_weights[workerid], decay_rate)
                for cat in self.CATEGORIES:
                    d["item_probs_" + cat] = self.compute_mode(final_weights, cat)
                
                for cat, v in d["answers"].items():
                    r = d["userResponse"][cat]
                    cat = cat.lower()
                    v = v.lower()
                    r = r.lower().strip()
                    likelihood += np.log(d["item_probs_" + cat][r]) if r in d["item_probs_" + cat] else 0
        return likelihood


    def compute_likelihood(self, N, attention_weights, decay_rate, workerid=None):
        likelihood = 0

        if workerid is not None:
            likelihood += self.compute_likelihood_for_worker(N, attention_weights, decay_rate, workerid)
        else:
            for wid in self.workerids:
                likelihood += self.compute_likelihood_for_worker(N, attention_weights, decay_rate, wid)


        return likelihood
    
    
    def make_sample(self, N, attention_weights, decay_rate):
        s =  {"N": N, "decay_rate": decay_rate}
        for wid in self.workerids:
            s[f"attention_weight_{wid}"] = attention_weights[wid]
        return s

    def compute_trans_probs(self, src_N, src_attention_weight, src_decay_rate, 
                          tgt_N, tgt_attention_weight, tgt_decay_rate):
      log_prob = 0
      if self.config["parameters"]["N"]["update"]:
        N_width = self.config["parameters"]["N"]["proposal_width"]
        src = np.log(src_N)
        tgt = np.log(tgt_N)
        log_prob += norm(src, N_width).logpdf(tgt)
      
      return log_prob
  
 
def run_mcmc():
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--out_path", type=str, help="Path to write the samples")
    parser.add_argument("--run", type=str, help="Name of the run")

    args = parser.parse_args()
    
    config_path = args.config
    model = KTTIDModel(args.out_path, args.run, config_path)
    
 
    acceptance = 0
    samples = []

    old_N, old_attention_weights, old_decay_rate = model.init_trace()

    N, attention_weights, decay_rate = old_N, copy.deepcopy(old_attention_weights), old_decay_rate
    old_likelihood = model.compute_likelihood(old_N, old_attention_weights, old_decay_rate)
    old_prior = model.prior(old_N, old_attention_weights, old_decay_rate)
    sample = model.make_sample(old_N, old_attention_weights, old_decay_rate)
    samples.append(sample)

    log_file_path = os.path.join(model.out_path, f"run{model.run}.log")
    samples_file_path = os.path.join(model.out_path, f"samples_run{model.run}.json")

    N_workerids = len(model.workerids)
    it_cycle_len = N_workerids + 1
    
    with open(log_file_path, "w", encoding="UTF-8") as log_file:
    
        for it in range(model.config["iterations"] * it_cycle_len):
            if it > 0 and it % 100 == 0:
                print("Iteration: {} ".format(it), file=log_file)
                print("Acceptance rate: {}".format(acceptance * 1.0 / it), file=log_file)
                print("Log likelihood: {}".format(old_likelihood), file=log_file)
                log_file.flush()
      
            cycle = it % it_cycle_len
            workerid = None
            if cycle == 0:
                N = model.proposal_N(old_N)
                decay_rate = model.proposal_decay_rate(old_decay_rate)
                #print(attention_weights)
                #print(old_attention_weights)
            else:
                workerid = model.workerids[cycle - 1]
                attention_weights[workerid] = model.proposal_attention_weight(old_attention_weights[workerid])

            if workerid is not None:
                likelihood_diff = model.compute_likelihood(old_N, attention_weights, old_decay_rate, workerid=workerid) - model.compute_likelihood(old_N, old_attention_weights, old_decay_rate, workerid=workerid)
#                print(f"DIFF: {likelihood_diff}" )
                new_likelihood = old_likelihood + likelihood_diff
                prior_diff = model.prior(old_N, attention_weights, old_decay_rate, workerid=workerid) -  model.prior(old_N, old_attention_weights, old_decay_rate, workerid=workerid)
                new_prior = old_prior + prior_diff
                #print(f"old_prior: {old_prior}")
 #               print(f"DIFF: {prior_diff}" )

            else:
                #print(f"old_likelihood: {old_likelihood}")
                #print(model.compute_likelihood(old_N, old_attention_weights, old_decay_rate))
                #print(f"old_prior: {old_prior}")
                new_likelihood =  model.compute_likelihood(N, old_attention_weights, decay_rate)
                new_prior = model.prior(N, old_attention_weights, decay_rate)
                #print(f"new_likelihood: {new_likelihood}")
                #print(f"new_prior: {new_prior}")

 #
 
            accept = (new_likelihood + new_prior) > (old_likelihood + old_prior)
            if not accept:
                fwd_prob = model.compute_trans_probs(old_N, old_attention_weights, old_decay_rate, N, attention_weights, decay_rate)
                bwd_prob = model.compute_trans_probs(N, attention_weights, decay_rate, old_N, old_attention_weights, old_decay_rate)
                likelihood_ratio = (new_likelihood + new_prior) - (old_likelihood + old_prior) - fwd_prob + bwd_prob
                u = np.log(uniform(0,1).rvs())
                if u < likelihood_ratio:
                    accept = True
                
            if accept:
                old_likelihood = new_likelihood
                old_prior = new_prior
                acceptance += 1
                if workerid is not None:
                    old_attention_weights[workerid] = attention_weights[workerid]
                    #print("updating AW")

                else:
                    #print("updating decay rate")
                    old_decay_rate = decay_rate
                    old_N = N

            
            if it > model.config["burn_in"] and it % (10 * it_cycle_len) == 0:
                sample = model.make_sample(old_N, old_attention_weights, old_decay_rate)
                samples.append(sample)
                if len(samples) % 1000 == 0:
                    with open(samples_file_path, "w", encoding="UTF-8") as out_f:
                        json.dump(samples, out_f)
    
    with open(samples_file_path, "w", encoding="UTF-8") as out_f:
        json.dump(samples, out_f)

if __name__ == "__main__":
    run_mcmc()
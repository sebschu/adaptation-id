import numpy as np
from scipy.stats import norm, uniform
import random
from collections import Counter
import csv, json
import sys


class KTTModel(object):

    CATEGORIES = {
        "animals": ["bear", "dog", "horse", "lion", "pig", "wolf"],
        "colors": ["blue", "green", "orange", "purple", "red", "yellow"],
        "countries": ["egypt", "brazil", "china", "france", "india", "japan"],
        "distances": ["meter", "mile", "inch", "kilometer", "foot", "yard"],
        "metals": ["tin", "nickel", "steel", "lead", "zinc", "iron"],
        "relatives": ["mother", "father", "brother", "sister", "aunt", "uncle"]
        }
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.load_data()
    
    def load_data(self):
        self.data = []
        with open(self.data_path, "r") as f:
            reader =  csv.DictReader(f)
            for line in reader:
                d = json.loads(json.loads(line["answer"]))
                self.data.append(d)


        

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
        
        decay_rate = random.uniform(0.8, 1.0)
        #N = random.normalvariate(150.0, 25.0)
        N = 150
        attention_weight = random.uniform(0.5, 1)
        
        return (N, attention_weight, decay_rate)

    def prior_N(self, old_val):
        return 150
        #width = 0.001
        #old_val = np.log(old_val)
        #return np.exp(norm(old_val, width).rvs())
        

    def prior_attention_weight(self, old_val):
        width = 0.001
        a = max(.5, old_val - width)
        b = min(1, old_val + width)
        b = max(0, b - a)
        return uniform(a, b).rvs()

    
    def prior_decay_rate(self, old_val):
        width = 0.001
        a = max(.5, old_val - width)
        b = min(1, old_val + width)
        b = max(0, b - a)
        return uniform(a, b).rvs()
    


        

    def compute_likelihood(self, N, attention_weight, decay_rate):
        

        def compute_mode(weights, cat):
            alphas, labels = weights[cat], self.CATEGORIES[cat]
            
            A = np.sum(alphas)
            xs = (alphas - 1) / (A - len(alphas))
            return dict(zip(labels, xs))

                
                

        #chosen_probs = []
        #gold_probs = []

        #correct_gold_probs = []
        #incorrect_gold_probs = []

        likelihood = 0
        counts = 0

        for d in self.data:
            final_weights = self.run_experiment(d, N, attention_weight, decay_rate)
            for cat in self.CATEGORIES:
                d["item_probs_" + cat] = compute_mode(final_weights, cat)
            
            #for cat, v in d["userResponse"].items():
            #    cat = cat.lower()
            #    v = v.lower()
                #if v in d["item_probs_" + cat]:
                #    chosen_probs.append(d["item_probs_" + cat][v])
                #else:
                #    chosen_probs.append(0)
                
            for cat, v in d["answers"].items():
                r = d["userResponse"][cat]
                cat = cat.lower()
                v = v.lower()
                r = r.lower().strip()
                #gold_probs.append(d["item_probs_" + cat][v])
                likelihood += np.log(d["item_probs_" + cat][r]) if r in d["item_probs_" + cat] else -3
                counts += 1
                #if v == r:
                #    correct_gold_probs.append(d["item_probs_" + cat][v])
                #else:
                #    incorrect_gold_probs.append(d["item_probs_" + cat][v])


        return likelihood
    
    
    def make_sample(self, N, attention_weight, decay_rate):
        return {"N": N, "attention_weight": attention_weight, "decay_rate": decay_rate}

    def compute_trans_probs(self, src_N, src_attention_weight, src_decay_rate, 
                          tgt_N, tgt_attention_weight, tgt_decay_rate):
      log_prob = 0
      #N_width = 0.001
      #other_width = 0.01
      #src = np.log(src_N)
      #tgt = np.log(tgt_N)
      #log_prob += norm(src, N_width).logpdf(tgt)
      
      return log_prob
  
 
def run_mcmc():
    data_path = "/Users/sebschu/Dropbox/Uni/RA/adaptation/individual-differences/experiments/5_talker_specific_adaptation_prior_exp_test_ktt/data/ktt_raw_HID.csv"
    model = KTTModel(data_path)
    
    burn_in = 0
    
    acceptance = 0
    samples = []

    old_N, old_attention_weight, old_decay_rate = model.init_trace()

    old_likelihood = model.compute_likelihood(old_N, old_attention_weight, old_decay_rate)
    sample = model.make_sample(old_N, old_attention_weight, old_decay_rate)

    samples.append(sample)
    
    log_file = sys.stderr
    
    for it in range(10000):
        if it > 0 and it % 100 == 0:
          print("Iteration: {} ".format(it), file=log_file)
          print("Acceptance rate: {}".format(acceptance * 1.0 / it), file=log_file)
          print("Log likelihood: {}".format(old_likelihood), file=log_file)
          
        N = model.prior_N(old_N)
        decay_rate = model.prior_decay_rate(old_decay_rate)
        attention_weight = model.prior_attention_weight(old_attention_weight)

        new_likelihood = model.compute_likelihood(N, attention_weight, decay_rate)
        accept = new_likelihood > old_likelihood
        if not accept:
          fwd_prob = model.compute_trans_probs(old_N, old_attention_weight, old_decay_rate, N, attention_weight, decay_rate)
          bwd_prob = model.compute_trans_probs(N, attention_weight, decay_rate, old_N, old_attention_weight, old_decay_rate)
          likelihood_ratio = new_likelihood - old_likelihood - fwd_prob + bwd_prob
          u = np.log(uniform(0,1).rvs())
          #print(likelihood_ratio, u)
          if u < likelihood_ratio:
            accept = True
        
        if accept:
          old_likelihood = new_likelihood
          sample = model.make_sample(N, attention_weight, decay_rate)
          acceptance += 1
        
        if it > burn_in and it % 10 == 0:
          samples.append(sample)
          if len(samples) % 10 == 0:
            json.dump(samples, open("samples.json", "w"))
  

    json.dump(samples, open("samples_final.json", "w"))




    
    
    

 
    
if __name__ == "__main__":
    run_mcmc()
import json
import argparse
import os


def beta_mu(a, b):
    return a / (a+b)


def beta_nu(a, b):
    return  a + b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--cautious_template", type=str, default="scripts/cautious_template.json")
    parser.add_argument("--confident_template", type=str, default="scripts/confident_template.json")
    parser.add_argument("--out_dir", type=str)
    
    
    
    args = parser.parse_args()
    
    mle_params = {}

    
    with open(args.input) as f:
        samples = json.load(f)
        param_sums = {k: 0.0 for k in samples[0].keys()}
        for s in samples:
            for k,v in s.items():
                param_sums[k] += v            

        n_samples = len(samples) * 1.0
        for k,v in param_sums.items():
            mle_params[k] = v / n_samples

    with open(args.cautious_template) as cau_t, open(args.confident_template) as con_t:
        cautious_config = json.load(cau_t)
        confident_config = json.load(con_t)
        
        # Rationality parameter
        cautious_config["rat_alpha_prior"]["rat_alpha_mu"] = mle_params["rat_alpha"]
        confident_config["rat_alpha_prior"]["rat_alpha_mu"] = mle_params["rat_alpha"]
        cautious_config["rat_alpha_init"] = mle_params["rat_alpha"]
        confident_config["rat_alpha_init"] = mle_params["rat_alpha"]

        # 'Other' probability
        cautious_config["utt_other_prob_prior"]["utt_other_prob_mu"] = mle_params["utt_other_prob"]
        confident_config["utt_other_prob_prior"]["utt_other_prob_mu"] = mle_params["utt_other_prob"]
        cautious_config["utt_other_prob_init"] = mle_params["utt_other_prob"]
        confident_config["utt_other_prob_init"] = mle_params["utt_other_prob"]
        
        # Noise strength
        cautious_config["noise_strength_init"] = mle_params["noise_strength"]
        confident_config["noise_strength_init"] = mle_params["noise_strength"]

        def update_utt(utt, mle_params):
            form = utt["form"]
            a = mle_params[f"alpha_{form}"]
            b = mle_params[f"beta_{form}"]
            cost = mle_params[f"cost_{form}"]
            mu = beta_mu(a,b)
            nu = beta_nu(a,b)
            utt["mu_init"] = mu
            utt["nu_init"] = nu
            utt["prior"]["mu_mu"] = mu
            if form != "might" and form != "probably":
                utt["prior"]["cost_mu"] = cost
                utt["cost_init"] = cost           


        for utt in cautious_config["utterances"]:
            update_utt(utt, mle_params)
        
        for utt in confident_config["utterances"]:
            update_utt(utt, mle_params)
        
                
        cautious_dir = os.path.join(args.out_dir, "cautious")
        confident_dir = os.path.join(args.out_dir, "cautious")

        os.makedirs(cautious_dir, exist_ok=True)
        os.makedirs(confident_dir, exist_ok=True)
        
        cautious_config_path = os.path.join(cautious_dir, "config.json")
        confident_config_path = os.path.join(confident_dir, "config.json")
        
        with open(cautious_config_path, "w") as cau_c, open(confident_config_path, "w") as con_c:
            json.dump(cautious_config, cau_c,  indent=2)
            json.dump(confident_config, con_c, indent=2)

        
if __name__ == "__main__":
    main()
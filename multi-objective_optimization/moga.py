# +
import random, multiprocessing, os, sys,torch,matplotlib
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors


from math import dist, e, log
from functools import partial
from deap import base, creator, tools, algorithms
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.spatial.distance import euclidean
import torch.nn.functional as F
matplotlib.rcParams['svg.fonttype'] = 'none'



class MOGA(BaseEstimator, TransformerMixin):
    def __init__(self, matching_result, patient_df, sec_fit, save_dir, incl_text, excl_text, model, tokenizer, sizePop = 100, numGen = 100, cxPb = 0.8, indMPb = 0.07, mutPb = 1, 
                crowding = 0.3, solutionSize = 1, min_active_rules = 15, seed=1, fitness_type = "enrollment+adverse"):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        self.matching_result, self.patient_df = matching_result, patient_df
        self.numGen = numGen
        self.cxPb = cxPb  
        self.indMPb = indMPb
        self.mutPb = mutPb 
        self.crowding = crowding
        self.sec_fit = sec_fit
        self.solutionSize = solutionSize
        self.log_path = os.path.join(save_dir, "MOGA_log.txt")
        self.save_dir = save_dir
        self.fitness_list = []
        self.toolbox = base.Toolbox()
        self.incl_text = incl_text
        self.excl_text = excl_text
        self.model = model
        self.tokenizer = tokenizer
        self.sizePop = sizePop
        self.min_active_rules = min_active_rules
        self.fitness_type = fitness_type
        
        for patient_id, patient_data in matching_result.items():
            incl_eval = patient_data.get("Inclusion Criteria Evaluation", [])
            excl_eval = patient_data.get("Exclusion Criteria Evaluation", [])
            break
        self.trial_length = len(incl_eval) + len(excl_eval)        
                
        if fitness_type == "enrollment+size":
            creator.create("Fitness", base.Fitness, weights=(1.0, 1.0))
        elif fitness_type == "enrollment+adverse":
            creator.create("Fitness", base.Fitness, weights=(1.0, -1.0))
        elif fitness_type == "enrollment+size+adverse":
            creator.create("Fitness", base.Fitness, weights=(1.0, 1.0, -1.0))
    
        creator.create("Individual", list, fitness=creator.Fitness)
        ref_points = tools.uniform_reference_points(len(creator.Fitness.weights), sizePop)
        
#         self.pool = multiprocessing.Pool(processes=1)
#         self.toolbox.register("map", self.pool.map)
#         self.toolbox.register("map", self.map)
        self.toolbox.register("attr_bool", random.randint, 0, 1)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_bool, self.trial_length)
        self.AE_full = self.compute_AE(creator.Individual([1] * self.trial_length))
        
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register(
            "evaluate",
            partial(
                fitness_dispatcher,
                fitness_type=self.fitness_type,     # <-- add this as constructor arg
                matching_result=self.matching_result,
                patient_df=self.patient_df,
                incl_text=self.incl_text,
                excl_text=self.excl_text,
                tokenizer=self.tokenizer,
                model=self.model,
                device="cuda",
                AE_full=self.AE_full,
            )
        )
        self.toolbox.register('mate', tools.cxOnePoint)
        self.toolbox.register('mutate', tools.mutFlipBit, indpb = indMPb)
        self.toolbox.register("NSGAselect", tools.selNSGA3, ref_points=ref_points)
#         self.toolbox.register("NSGAselect", tools.selNSGA2)

    def compute_AE(self, ind):
        # Mask rules
        n_incl = len(self.incl_text)
        active_incl_rules = [r for bit, r in zip(ind[:n_incl], self.incl_text) if bit == 1]
        active_excl_rules = [r for bit, r in zip(ind[n_incl:], self.excl_text) if bit == 1]

        inc_full = "\n".join(["inclusion criteria:"] + active_incl_rules)
        exc_full = "\n".join(["exclusion criteria:"] + active_excl_rules)

        enc = self.tokenizer(
            inc_full,
            text_pair=exc_full,
            truncation=True,
            padding="longest",
            max_length=512,
            return_tensors="pt"
        )
        enc = {k: v.to("cuda") for k, v in enc.items()}

        with torch.no_grad():
            logits = self.model(**enc).logits
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()  # P(AE)

        return float(probs.mean())

    def fit(self, X=None, y=None):
        return self
    
    def transform(self, X=None, y=None):
        '''
        Evolution starts
        '''
        with open(self.log_path, "a", encoding="utf-8") as f:
            
            pop = self.toolbox.population(n=self.sizePop)
            fitnesses = list(self.toolbox.map(self.toolbox.evaluate, pop))
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit

            for gen in range(1, self.numGen + 1):
                print(f"\n-- Generation {gen} --")
                f.write(f"\n-- Generation {gen} --\n")

                offspring = list(map(self.toolbox.clone, pop))
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < self.cxPb:
                        self.toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values

                for mutant in offspring:
                    if random.random() < self.mutPb:
                        self.toolbox.mutate(mutant)
                        del mutant.fitness.values
                        
                for ind in offspring:
                    while sum(ind) < self.min_active_rules:
                        on_indices = np.random.choice(len(ind), self.min_active_rules, replace=False)
                        for i in on_indices:
                            ind[i] = 1
                        
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = list(self.toolbox.map(self.toolbox.evaluate, invalid_ind))
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                pop = self.toolbox.NSGAselect(pop + offspring, self.sizePop)
                
                if gen == 1:
                    all_ones = creator.Individual([1] * self.trial_length)
                    pop[0] = all_ones
                    all_ones.fitness.values = self.toolbox.evaluate(all_ones)
                    print(f"Full Rule Fitness: {all_ones.fitness.values}")
                    pop[0] = all_ones
        
                fits = np.array([ind.fitness.values for ind in pop])
                self.plot_fitness_scatter(fits, gen)
                self.save_generation_details(pop, gen)
                
                gen_stats = []  # Store max, min, and avg for each objective in this generation
                for i in range(fits.shape[1]):
                    max_fit = np.max(fits[:, i])
                    min_fit = np.min(fits[:, i])
                    avg_fit = np.mean(fits[:, i])
                    gen_stats.append((max_fit, min_fit, avg_fit))
                    print(f" F{i+1}: Avg = {avg_fit:.5f}, Max = {max_fit:.5f}, Min = {min_fit:.5f}")
                    f.write(f" F{i+1}: Avg = {avg_fit:.5f}, Max = {max_fit:.5f}, Min = {min_fit:.5f}\n")
                self.fitness_list.append(gen_stats)  # Append the stats for this generation

            sorted_pop = sorted(pop, key=lambda ind: ind.fitness.values[0], reverse=True) 
            optimal_solutions = sorted_pop[:self.solutionSize]
            self.optimal_solutions = pop
#             self.plot_fitness_evolution()
        return optimal_solutions
    
    def save_generation_details(self, population, gen):
        """
        Save all individuals of a generation, including decoded rules,
        fitness scores (any number of objectives), rule counts,
        and eligible patient IDs.
        """
        records = []
        gen_dir = os.path.join(self.save_dir, "evolution_records")
        os.makedirs(gen_dir, exist_ok=True)
        save_path = os.path.join(gen_dir, f"generation_{gen:03d}.csv")

        for i, ind in enumerate(population):
            decoded = self.decode_solution(ind)
            fit_values = list(ind.fitness.values)      # dynamic

            # ---- Rule counts ----
            num_rules_total = int(len(ind))

            # ---- Eligible patient IDs ----
            eligible_patient_ids = []
            for patient_id, patient_data in self.matching_result.items():
                incl_eval = patient_data.get("Inclusion Criteria Evaluation", [])
                excl_eval = patient_data.get("Exclusion Criteria Evaluation", [])

                inclusion_decisions = [entry.split("**")[-1] for entry in incl_eval]
                exclusion_decisions = [entry.split("**")[-1] for entry in excl_eval]

                masked_inclusions = [
                    d for bit, d in zip(ind[:len(inclusion_decisions)], inclusion_decisions)
                    if bit == 1
                ]
                masked_exclusions = [
                    d for bit, d in zip(ind[len(inclusion_decisions):], exclusion_decisions)
                    if bit == 1
                ]

                if len(masked_inclusions) == 0 and len(masked_exclusions) == 0:
                    eligible_patient_ids.append(patient_id)
                elif ("yes" not in masked_inclusions or 
                      "no" in masked_inclusions or 
                      "yes" in masked_exclusions):
                    continue
                else:
                    eligible_patient_ids.append(patient_id)
        
            rec = {}
            for idx, f in enumerate(fit_values):
                rec[f"fitness_{idx+1}"] = f

            # ---- THEN: Metadata columns ----
            rec.update({
                "generation": gen,
                "individual_id": i,
                "num_rules_total": num_rules_total,
                "eligible_patient_ids": ",".join(map(str, eligible_patient_ids)),
                "inclusion_rules": " | ".join(decoded["inclusion_rules"]),
                "exclusion_rules": " | ".join(decoded["exclusion_rules"]),
            })

            records.append(rec)

        df = pd.DataFrame(records)
        df.to_csv(save_path, index=False)
        print(f"📈 Saved generation {gen} details to {save_path}")
    
    
    def decode_solution(self, individual):
        """
        Map each bit in the binary vector back to its rule text.
        Returns a dictionary with active inclusion and exclusion rules.
        """
        n_incl = len(self.incl_text)
        active_incl_rules = [r for bit, r in zip(individual[:n_incl], self.incl_text) if bit == 1]
        active_excl_rules = [r for bit, r in zip(individual[n_incl:], self.excl_text) if bit == 1]

        return {
            "inclusion_rules": active_incl_rules,
            "exclusion_rules": active_excl_rules
        }

#     def plot_fitness_evolution(self):
#         num_generations = len(self.fitness_list)
#         num_objectives = len(self.fitness_list[0])

#         # Create a figure with subplots for each fitness objective
#         fig, axs = plt.subplots(num_objectives, 1, figsize=(10, num_objectives * 5), squeeze=False)
#         axs = axs.flatten()  # Flatten in case there's only one subplot

#         # Set a title for the whole figure
#         fig.suptitle('Evolution of Fitness Objectives Over Generations', fontsize=16)

#         # Iterate over each fitness objective
#         for i in range(num_objectives):
#             max_values = [generation[i][0] for generation in self.fitness_list]  # Extract max values for this objective
#             min_values = [generation[i][1] for generation in self.fitness_list]  # Extract min values for this objective
#             avg_values = [generation[i][2] for generation in self.fitness_list]  # Extract avg values for this objective

#             generations = range(1, num_generations + 1)
#             axs[i].plot(generations, max_values, label='Max', marker='o', linestyle='-', color='r')
#             axs[i].plot(generations, min_values, label='Min', marker='x', linestyle='--', color='b')
#             axs[i].plot(generations, avg_values, label='Avg', marker='s', linestyle='-.', color='g')

#             axs[i].set_title(f'Objective {i+1}')
#             axs[i].set_xlabel('Generation')
#             axs[i].set_ylabel('Fitness Value')
#             axs[i].legend()
#             axs[i].grid(True)

#         # Adjust layout for better spacing
#         plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#         plt.savefig(os.path.join(self.save_dir,"fitness_evolution.png"))
#         plt.savefig(os.path.join(self.save_dir,"fitness_evolution.svg"))
        
        
    def plot_fitness_scatter(self, fitness_list, gen):
        num_dimensions = len(fitness_list[0])
        df = pd.DataFrame(fitness_list)

        # ----------------------------------------------
        # 1-D scatter (rare, but supporting for completeness)
        # ----------------------------------------------
        if num_dimensions == 1:
            fig = plt.figure(figsize=(6, 4))
            plt.scatter(range(len(df)), df.iloc[:, 0], s=80)
            plt.xlabel("Individual")
            plt.ylabel("Fitness 1")
            plt.title(f"Generation {gen}: Fitness 1 Scatter")

        # ----------------------------------------------
        # 2-D scatter plot
        # ----------------------------------------------
        elif num_dimensions == 2:
            df.columns = ['Fitness 1', 'Fitness 2']
            plt.figure(figsize=(6, 5))
            sns.scatterplot(data=df, x='Fitness 1', y='Fitness 2', s=80)

            plt.title(f"Generation {gen}: 2D Fitness Scatter")
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
            if self.fitness_type == "enrollment+size":
                ax.set_xlim(0, 2500)
                ax.set_ylim(0, 40)

#             elif self.fitness_type == "enrollment+adverse":
#                 ax.set_xlim(0, 1550)
#                 ax.set_ylim(0.65, 1)
                
                
        # ----------------------------------------------
        # 3-D scatter plot
        # ----------------------------------------------
        elif num_dimensions == 3:
            df.columns = ['Fitness 1', 'Fitness 2', 'Fitness 3']

            plt.figure(figsize=(7, 6))
            sc = plt.scatter(
                df['Fitness 1'],
                df['Fitness 3'],
                c=df['Fitness 2'],      # color = 3rd fitness
                cmap=mcolors.LinearSegmentedColormap.from_list("redWhiteBlue",["#2878B5", "#FFFFFF", "#D44A4A"]),   # blue → white → red
                s=80,
                alpha=0.9,
                edgecolor='k'
            )

            cbar = plt.colorbar(sc)
            cbar.set_label("Fitness 3 (Color Scale)")

            plt.xlabel("Fitness 1")
            plt.ylabel("Fitness 3")
            plt.title(f"Generation {gen}: 2D Scatter (Fitness 3 as Color)")

            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlim(0,2500)
            ax.set_ylim(0,1)
            
        save_path = os.path.join(self.save_dir, "fitnesses") 
        os.makedirs(save_path, exist_ok=True) 
        plt.savefig(os.path.join(save_path, f"fitness_{num_dimensions}d_gen{gen}.png")) 
        plt.savefig(os.path.join(save_path, f"fitness_{num_dimensions}d_gen{gen}.svg")) 
        plt.close()
        

def fitness_dispatcher(ind, fitness_type,
                       matching_result, patient_df,
                       incl_text, excl_text,
                       tokenizer, model, device,
                       AE_full):
    """
    Dispatches to different fitness functions based on fitness_type.

    Supported:
        - "enrollment + adverse"
        - "enrollment + size"
        - "enrollment + size + adverse"

    Returns a fitness tuple, for DEAP NSGA.
    """

    # ---- 1️⃣ Enrollment counting (always needed) ----
    num_eligible = 0

    for patient_id, patient_data in matching_result.items():
        incl_eval = patient_data.get("Inclusion Criteria Evaluation", [])
        excl_eval = patient_data.get("Exclusion Criteria Evaluation", [])

        inclusion_decisions = [entry.split("**")[-1] for entry in incl_eval]
        exclusion_decisions = [entry.split("**")[-1] for entry in excl_eval]

        masked_inclusions = [d for bit, d in zip(ind[:len(inclusion_decisions)], inclusion_decisions) if bit == 1]
        masked_exclusions = [d for bit, d in zip(ind[len(inclusion_decisions):], exclusion_decisions) if bit == 1]

        # Eligibility logic
        if len(masked_inclusions) == 0 and len(masked_exclusions) == 0:
            num_eligible += 1
        elif "yes" not in masked_inclusions or "no" in masked_inclusions or "yes" in masked_exclusions:
            continue
        else:
            num_eligible += 1

    # ---- 2️⃣ Rule size (always computed if needed) ----
    total_rules = len(ind)
    num_rules_selected = sum(ind)

    # ---- 3️⃣ AE computation (only if needed) ----
    if "adverse" in fitness_type:
        # Build selected rule text
        n_incl = len(incl_text)
        active_incl_rules = [r for bit, r in zip(ind[:n_incl], incl_text) if bit == 1]
        active_excl_rules = [r for bit, r in zip(ind[n_incl:], excl_text) if bit == 1]

        inc_full = "\n".join(["inclusion criteria:"] + active_incl_rules)
        exc_full = "\n".join(["exclusion criteria:"] + active_excl_rules)

        enc = tokenizer(
            inc_full, text_pair=exc_full,
            truncation=True, padding="longest",
            max_length=512, return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits
            AE_new = float(F.softmax(logits, dim=1)[:, 1].cpu().numpy().mean())

#         # AE penalty (your monotonic rule)
#         if AE_new < AE_full:
#             AE_penalty = 1e3
#         else:
#             AE_penalty = AE_new - AE_full
#     else:
#         AE_penalty = None  # not used

    # ---- 4️⃣ Dispatch based on fitness_type ----

    if fitness_type == "enrollment+adverse":
        return (float(num_eligible), float(AE_new))

    elif fitness_type == "enrollment+size":
        # maximize enrollment, minimize num_rules_removed
        return (float(num_eligible), float(num_rules_selected))

    elif fitness_type == "enrollment+size+adverse":
        return (float(num_eligible), float(num_rules_selected), float(AE_new))

    else:
        raise ValueError(f"Unknown fitness_type: {fitness_type}")
        
def fitness_enrollment_mortality(
        ind, matching_result, patient_df, incl_text, excl_text,
        tokenizer, model, device, AE_full
    ):
    """
    Objective 1: maximize number of eligible patients
    Objective 2: minimize safety penalty:
         If AE_new < AE_full  -> huge penalty (invalid)
         If AE_new = AE_full  -> 0
         If AE_new > AE_full  -> AE_new - AE_full
    """

    # ---- (1) Count eligible patients ----
    num_eligible = 0
    for patient_id, patient_data in matching_result.items():
        incl_eval = patient_data.get("Inclusion Criteria Evaluation", [])
        excl_eval = patient_data.get("Exclusion Criteria Evaluation", [])

        inclusion_decisions = [entry.split("**")[-1] for entry in incl_eval]
        exclusion_decisions = [entry.split("**")[-1] for entry in excl_eval]

        masked_inclusions = [d for bit, d in zip(ind[:len(inclusion_decisions)], inclusion_decisions) if bit == 1]
        masked_exclusions = [d for bit, d in zip(ind[len(inclusion_decisions):], exclusion_decisions) if bit == 1]

        if len(masked_inclusions) == 0 and len(masked_exclusions) == 0:
            num_eligible += 1
        elif "yes" not in masked_inclusions or "no" in masked_inclusions or "yes" in masked_exclusions:
            continue
        else:
            num_eligible += 1

    # ---- (2) Compute AE_new for THIS individual ----
    n_incl = len(incl_text)
    active_incl_rules = [r for bit, r in zip(ind[:n_incl], incl_text) if bit == 1]
    active_excl_rules = [r for bit, r in zip(ind[n_incl:], excl_text) if bit == 1]

    inc_full = "\n".join(["inclusion criteria:"] + active_incl_rules)
    exc_full = "\n".join(["exclusion criteria:"] + active_excl_rules)

    enc = tokenizer(
        inc_full, text_pair=exc_full,
        truncation=True, padding="longest",
        max_length=512, return_tensors="pt"
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits
        AE_new = float(F.softmax(logits, dim=1)[:, 1].cpu().numpy().mean())

    # ---- (3) Safety penalty following YOUR rule ----
    if AE_new < AE_full:
        penalty = 1e6   # huge penalty
    else:
        penalty = AE_new - AE_full          # normal penalty

    return (float(num_eligible), float(penalty))

   


    

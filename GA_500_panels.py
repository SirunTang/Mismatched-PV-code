import pandas as pd
import numpy as np
import math
# === Load panel data ===
def load_panels(filepath):
    df = pd.read_csv(filepath, low_memory=False)
    df = df.dropna(subset=['UOC', 'ISC', 'UMPP', 'IMPP'])

    panels = []
    for idx, row in df.iterrows():
        panel = {
            'id': len(panels),
            'Voc': float(row['UOC']),
            'Isc': float(row['ISC']),
            'Vmp': float(row['UMPP']),
            'Imp': float(row['IMPP']),
        }
        panels.append(panel)

    return panels  # Required return value

panels = load_panels("dataset1.csv")
print(f"success {len(panels)} numbers")
print(panels[0])

import numpy as np
import random


import random

# === Initialize variable-length population ===
def initialize_population_variable_L(panels, G, min_L=6, max_L=12, population_size=20, imp_tol=0.10):
    """
    Initialize population. Each individual consists of G groups × 5 strings.
    Each string has length L ∈ [min_L, max_L] satisfying:
    - Total string voltage: 360V ≤ sum(Vmp) ≤ 400V
    - Current consistency: max(Imp) ≤ min(Imp) × (1 + imp_tol)
    Only valid strings are retained.
    """
    total_strings = G * 5
    population = []

    for pop_idx in range(population_size):
        panels_pool = panels.copy()
        random.shuffle(panels_pool)
        panels_pool.sort(key=lambda p: p['Imp'])

        individual = []
        used_ids = set()
        attempts = 0
        max_attempts = 10000

        while len(individual) < total_strings and attempts < max_attempts:
            success = False
            for start_idx in range(len(panels_pool) - min_L + 1):
                for L in range(min_L, max_L + 1):
                    if start_idx + L > len(panels_pool):
                        continue
                    candidate = panels_pool[start_idx:start_idx + L]
                    if any(p['id'] in used_ids for p in candidate):
                        continue
                    V_sum = sum(p['Vmp'] for p in candidate)
                    Imp_list = [p['Imp'] for p in candidate]
                    if 360 <= V_sum <= 400 and max(Imp_list) <= min(Imp_list) * (1 + imp_tol):
                        individual.append(candidate)
                        for p in candidate:
                            used_ids.add(p['id'])
                        success = True
                        break
                if success:
                    break
            if not success:
                attempts += 1
                random.shuffle(panels_pool)


        valid_count = len(individual)
        print(f" Valid strings extracted = {valid_count}")
        population.append(individual)

    print(f" Final initial population count = {len(population)} （target = {population_size}）")
    return population





def evaluate_fitness_lm(individual):
    """
    Calculate the total output power of an individual:
    Each string contributes: min(Imp) * sum(Vmp)
    Apply penalties based on:
    - Voltage variation across strings
    - Isc variation within each string
    """
    total_power = 0
    total_voltages = []
    isc_std_total = 0

    for string in individual:
        currents = [p['Imp'] for p in string]
        voltages = [p['Vmp'] for p in string]
        iscs = [p['Isc'] for p in string]

        min_current = min(currents)
        total_voltage = sum(voltages)
        total_power += min_current * total_voltage

        total_voltages.append(total_voltage)
        isc_std_total += np.std(iscs)


    voltage_std = np.std(total_voltages)

    # Penalty control factors
    alpha = 3.0  # voltage consistency
    beta = 1.0   # Isc consistency

    penalty = alpha * voltage_std + beta * isc_std_total
    total_power -= penalty

    return total_power

# === Group-based fitness evaluation ===

def evaluate_fitness_by_groups(individual, group_size=5, voltage_limit=400, imp_tol=0.10, voltage_tol=0.10):
    """
        Evaluate the fitness of an individual based on G groups × 5 strings:
        - Validate each string's voltage and Imp consistency
        - Validate voltage consistency across parallel strings
        - Calculate total power as line_current × total string voltage per group
    """

    total_power = 0
    penalty = 0
    total_groups = len(individual) // group_size

    for g in range(total_groups):
        group = individual[g * group_size: (g + 1) * group_size]
        group_voltages = []
        valid = True

        for s_idx, string in enumerate(group):
            voltages = [p['Vmp'] for p in string]
            currents = [p['Imp'] for p in string]
            total_voltage = sum(voltages)
            min_current = min(currents)

            if total_voltage > voltage_limit:
                print(f" Voltage exceeds limit（{total_voltage:.2f} V），group {g} string {s_idx}")
                valid = False
                break

            for imp in currents:
                if imp > min_current * (1 + imp_tol):
                    print(f" Imp inconsistency（{imp:.2f} A vs min {min_current:.2f} A），group {g} string {s_idx}")
                    valid = False
                    break

            group_voltages.append(total_voltage)

        if not valid:
            penalty += 1
            continue

        max_v = max(group_voltages)
        min_v = min(group_voltages)
        if (max_v - min_v) / max_v > voltage_tol:
            print(f" Parallel voltage inconsistency（{max_v:.2f} vs {min_v:.2f}），group {g}")
            penalty += 1
            continue

        line_current = min(min(p['Imp'] for p in string) for string in group)
        group_power = sum(sum(p['Vmp'] for p in string) for string in group) * line_current
        total_power += group_power

    return total_power - penalty * 1e6




def evaluate_random_baseline(panels, L, M, num_trials=30):

    baseline_scores = []
    total_required = L * M
    for _ in range(num_trials):
        shuffled = random.sample(panels, total_required)
        individual = [shuffled[i*L:(i+1)*L] for i in range(M)]
        score = evaluate_fitness_lm(individual)
        baseline_scores.append(score)

    avg_score = sum(baseline_scores) / len(baseline_scores)
    max_score = max(baseline_scores)
    min_score = min(baseline_scores)
    return avg_score, max_score, min_score

def save_final_voltage_current_summary(individual, filename="final_voltage_current_summary.csv"):
    """
    Save line voltage (sum of Vmp) and line current (minimum Imp) for each string.
    """
    import csv
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['String Index', 'Line Voltage (V)', 'Line Current (A)', 'Power (W)'])
        for i, string in enumerate(individual):
            total_voltage = sum(p['Vmp'] for p in string)
            line_current = min(p['Imp'] for p in string)
            power = total_voltage * line_current
            writer.writerow([i, total_voltage, line_current, power])

def tournament_selection(population, scores, k=3):
    """
    Tournament selection: randomly select k individuals and return the one with the highest fitness.

    Args:
        population: list of all individuals
        scores: list of fitness scores for individuals
        k: tournament size (number of individuals to compare)

    Returns:
        Selected individual (used for crossover)
    """
    selected = random.sample(list(zip(population, scores)), k)
    selected.sort(key=lambda x: x[1], reverse=True)  # 按适应度从高到低排序
    return selected[0][0]

def crossover(parent1, parent2):
    """
    Single-point crossover: split and combine at string level.
    Note: does not shuffle panels or strings, only splices strings.
    """
    point = random.randint(1, len(parent1) - 1)
    child = parent1[:point] + parent2[point:]
    return child


def ensure_unique(individual, all_panels):
    used_ids = set()
    all_ids = set(p['id'] for p in all_panels)

    for string in individual:
        for idx, panel in enumerate(string):
            pid = panel['id']
            if pid in used_ids:
                available_ids = list(all_ids - used_ids)
                if not available_ids:
                    print("⚠️ Warning: no available panels to replace duplicate ID. Panel kept.")
                    continue
                candidates = [p for p in all_panels if p['id'] in available_ids]
                new_panel = random.choice(candidates)
                string[idx] = new_panel
                used_ids.add(new_panel['id'])
            else:
                used_ids.add(pid)

    return individual



def mutate(individual, mutation_rate, all_panels):
    """
    Mutation: 对每个 string 中的每块面板，以 mutation_rate 的概率
    用一个尚未在个体中出现的面板替换，保持串长度不变。
    """
    import copy
    mutated = copy.deepcopy(individual)
    # 先收集 mutated 中所有已用面板的 id
    used_ids = {p['id'] for string in mutated for p in string}

    for string in mutated:
        for idx in range(len(string)):
            if random.random() < mutation_rate:
                # 从剩余未用的面板里随机选一个
                candidates = [p for p in all_panels if p['id'] not in used_ids]
                if not candidates:
                    # 全部面板都用过了，跳过这次替换
                    continue
                replacement = random.choice(candidates)
                # 更新 used_ids：移除被替换的旧 id，加入新面板 id
                old_id = string[idx]['id']
                used_ids.discard(old_id)
                used_ids.add(replacement['id'])
                string[idx] = replacement

    return mutated


def run_genetic_algorithm(panels, G, generations=50, population_size=20, mutation_rate=0.2):
    """
    Main workflow of the genetic algorithm.
    Each individual consists of G groups × 5 strings, with variable-length strings.
    """
    population = initialize_population_variable_L(panels, G, population_size=population_size)
    best_individual = None
    best_fitness = -1

    for gen in range(generations):
        fitnesses = [evaluate_fitness_by_groups(ind) for ind in population]

        elite_idx = np.argmax(fitnesses)
        elite = population[elite_idx]
        elite_fitness = fitnesses[elite_idx]

        if elite_fitness > best_fitness:
            best_fitness = elite_fitness
            best_individual = elite

        print(f"Generation {gen+1:02d} : Best fitness ={elite_fitness:.2f} W")

        new_population = [elite]

        while len(new_population) < population_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child = crossover(parent1, parent2)
            if random.random() < mutation_rate:
                child = mutate(child, mutation_rate, all_panels=panels)

            # Check if all strings in child meet Imp restriction
            valid = True
            for string in child:
                imp_list = [p['Imp'] for p in string]
                min_imp = min(imp_list)
                if any(imp > min_imp * 1.10 for imp in imp_list):
                    valid = False
                    break

            if valid:
                new_population.append(child)
            # else: discard the child silently

        population = new_population

    print("\n Genetic algorithm completed")
    print(f"Best total power = {best_fitness:.2f} W")
    return best_individual, best_fitness

# === Save Power Summary to TXT ===
def generate_power_summary_txt(best_individual, avg_random, max_random, min_random, best_score, filename="power_summary_report.txt"):
    group_powers = []
    lines = []

    lines.append("=== Group-wise Power Output ===")
    total_groups = math.ceil(len(best_individual) / 5)
    for g in range(total_groups):
        group = best_individual[g * 5: (g + 1) * 5]
        line_current = min(min(p['Imp'] for p in string) for string in group)
        group_voltage = sum(sum(p['Vmp'] for p in string) for string in group)
        power = line_current * group_voltage
        group_powers.append(power)
        if len(group) < 5:
            lines.append(f"Group {g:02d} (incomplete): Power = {power:.2f} W")
        else:
            lines.append(f"Group {g:02d}: Power = {power:.2f} W")

    avg_group_power = sum(group_powers) / len(group_powers)
    max_group_power = max(group_powers)
    min_group_power = min(group_powers)

    lines.append("\n=== Summary of Optimized Layout ===")
    total_from_groups = sum(group_powers)
    lines.append(f"Total Power       = {total_from_groups:.2f} W")

    lines.append(f"Average Group     = {avg_group_power:.2f} W")
    lines.append(f"Max Group Power   = {max_group_power:.2f} W")
    lines.append(f"Min Group Power   = {min_group_power:.2f} W")

    lines.append("\n=== Random Layout Benchmark ===")
    lines.append(f"Average Random    = {avg_random:.2f} W")
    lines.append(f"Max Random        = {max_random:.2f} W")
    lines.append(f"Min Random        = {min_random:.2f} W")

    improvement = best_score - avg_random
    improvement_percent = (best_score / avg_random - 1) * 100
    lines.append(f"\n Improvement = {improvement:.2f} W ({improvement_percent:.2f}%)")

    # Print to console
    print("\n".join(lines))

    # Save to TXT file
    with open(filename, mode='w') as f:
        f.write("\n".join(lines))


if __name__ == "__main__":

    # Load panel data
    panels = load_panels("dataset1.csv")
    print(f"Successfully loaded {len(panels)} panels")

    # Estimate how many groups G can be formed (each group: 5 strings, ~10 panels per string)
    G = len(panels) // (5 * 10)
    print(f"Estimated {G} groups can be formed, each group contains 5 strings")

    # Run genetic algorithm
    best_individual, best_score = run_genetic_algorithm(
        panels, G=G,
        generations=30,
        population_size=20,
        mutation_rate=0.1
    )

    print(f"\n Genetic algorithm completed. Best power = {best_score:.2f} W")



    def save_individual_to_csv(individual, filename="best_individual.csv"):
        """
        Save the best individual. Each group contains 5 strings.
        Format: group, string, position, ID, Voc, Isc, Vmp, Imp
        """
        import csv
        with open(filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Group', 'String', 'Position', 'ID', 'Voc', 'Isc', 'Vmp', 'Imp'])
            for s_idx, string in enumerate(individual):
                group = s_idx // 5
                for j, panel in enumerate(string):
                    writer.writerow([
                        group, s_idx % 5, j,
                        panel['id'], panel['Voc'], panel['Isc'], panel['Vmp'], panel['Imp']
                    ])


    def save_final_voltage_current_summary(individual, filename="final_voltage_current_summary.csv"):
        """
        Save the line voltage (sum of Vmp), line current (min Imp), and total power for each string in each group.
        """
        import csv
        with open(filename, mode='w', encoding='utf-8') as f:

            writer = csv.writer(f)
            writer.writerow(['Group', 'String', 'Voltage (V)', 'Current (A)', 'Power (W)'])
            for s_idx, string in enumerate(individual):
                group = s_idx // 5
                total_voltage = sum(p['Vmp'] for p in string)
                line_current = min(p['Imp'] for p in string)
                power = total_voltage * line_current
                writer.writerow([group, s_idx % 5, total_voltage, line_current, power])



# === Power Benchmark Comparison ===
L_mean = int(np.mean([len(s) for s in best_individual]))
M_total = len(best_individual)
print(f"\n Power comparison parameters: L = {L_mean}, M = {M_total}")

avg_random, max_random, min_random = evaluate_random_baseline(
    panels, L=L_mean, M=M_total, num_trials=30
)

save_individual_to_csv(best_individual, "best_individual.csv")
save_final_voltage_current_summary(best_individual, "final_voltage_current_summary.csv")
generate_power_summary_txt(best_individual, avg_random, max_random, min_random, best_score)

print("\n Random arrangement baseline (average {} panels per string):".format(L_mean))
print(f"Average power = {avg_random:.2f} W")
print(f"Maximum power = {max_random:.2f} W")
print(f"Minimum power = {min_random:.2f} W")
print(f"⚡ Power improvement after optimization = {(best_score - avg_random):.2f} W ({(best_score / avg_random - 1)*100:.1f}%)")


import pandas as pd

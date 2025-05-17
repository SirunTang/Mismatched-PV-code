import pandas as pd

# === Load both files ===
best_individual_df = pd.read_csv("best_individual.csv")
dataset_df = pd.read_csv("dataset1.csv")

# === Initialize a list to store verification results ===
verification_results = []

# === Iterate through each panel in best_individual ===
for _, row in best_individual_df.iterrows():
    idx = int(row['ID'])  # The ID is treated as the row index in dataset1
    dataset_row = dataset_df.iloc[idx]

    # Extract voltage and current values from both sources
    voc_best = row['Voc']
    isc_best = row['Isc']
    vmp_best = row['Vmp']
    imp_best = row['Imp']

    voc_data = dataset_row['UOC']
    isc_data = dataset_row['ISC']
    vmp_data = dataset_row['UMPP']
    imp_data = dataset_row['IMPP']

    # Compare values with a small tolerance
    voc_match = abs(voc_best - voc_data) < 1e-3
    isc_match = abs(isc_best - isc_data) < 1e-3
    vmp_match = abs(vmp_best - vmp_data) < 1e-3

    import pandas as pd

    # === Load best individual and dataset ===
    best_individual_df = pd.read_csv("best_individual.csv")
    dataset_df = pd.read_csv("dataset1.csv")


    # === 1. String-Level Voltage and Current Restriction Check ===
    def check_string_restrictions(individual_df, dataset_df, voltage_min=360, voltage_max=400, imp_tol=0.10):
        results = []
        grouped = individual_df.groupby(['Group', 'String'])

        for (g, s), string_df in grouped:
            panel_ids = string_df['ID'].values
            panels = dataset_df.iloc[panel_ids]

            total_voltage = panels['UMPP'].sum()
            imp_values = panels['IMPP'].values
            min_imp = imp_values.min()
            max_imp = imp_values.max()
            imp_error_pct = ((max_imp - min_imp) / max_imp * 100) if max_imp != 0 else 0.0

            voltage_ok = voltage_min <= total_voltage <= voltage_max
            imp_ok = all(imp <= min_imp * (1 + imp_tol) for imp in imp_values)

            results.append({
                'Group': g,
                'String': s,
                'Voltage(V)': round(total_voltage, 2),
                'Min_Imp(A)': round(min_imp, 2),
                'Max_Imp(A)': round(max_imp, 2),
                'Imp Range (A)': round(max_imp - min_imp, 2),
                'Imp Error (%)': round(imp_error_pct, 2),
                'Voltage_OK': voltage_ok,
                'Imp_OK': imp_ok
            })

        df = pd.DataFrame(results)
        print("=== String-Level Restriction Check ===")
        print(df.to_string(index=False))
        return df


    # === 2. Group-Level Parallel Voltage Consistency Check ===
    def check_group_parallel_voltage(individual_df, dataset_df, voltage_tol=0.10, group_size=5):
        group_results = []
        grouped = individual_df.groupby(['Group', 'String'])

        # Compute voltage per string
        string_voltages = {}
        for (g, s), string_df in grouped:
            panel_ids = string_df['ID'].values
            panels = dataset_df.iloc[panel_ids]
            total_voltage = panels['UMPP'].sum()
            string_voltages[(g, s)] = total_voltage

        # Evaluate all groups including incomplete ones
        max_group = individual_df['Group'].max()
        for g in range(max_group + 1):
            voltages = [v for (grp, s), v in string_voltages.items() if grp == g]
            if len(voltages) == 0:
                continue  # skip empty group

            max_v = max(voltages)
            min_v = min(voltages)
            relative_diff = (max_v - min_v) / max_v
            ok = relative_diff <= voltage_tol

            # ✅ Compute additional metrics
            avg_v = sum(voltages) / len(voltages)
            std_v = pd.Series(voltages).std()
            range_v = max_v - min_v

            group_results.append({
                'Group': g,
                'Max_Voltage': round(max_v, 2),
                'Min_Voltage': round(min_v, 2),
                'Avg_Voltage': round(avg_v, 2),  # ✅ new
                'Voltage_Std(V)': round(std_v, 2),  # ✅ new
                'Voltage_Range(V)': round(range_v, 2),  # ✅ new
                'Relative_Diff(%)': round(relative_diff * 100, 2),
                'Parallel_OK': ok,
                'Note': "Incomplete group" if len(voltages) < group_size else "Complete group"
            })

        return pd.DataFrame(group_results)


    # === Run both checks ===
    string_df = check_string_restrictions(best_individual_df, dataset_df)
    group_df = check_group_parallel_voltage(best_individual_df, dataset_df)


# === Final verification summary ===
used_ids = set(best_individual_df['ID'].unique())
all_ids = set(dataset_df.index)

unused_ids = all_ids - used_ids
num_used = len(used_ids)
num_total = len(all_ids)
num_unused = len(unused_ids)
num_failed_voltage = sum(~string_df["Voltage_OK"])
num_failed_imp = sum(~string_df["Imp_OK"])
num_failed_parallel = sum(~group_df["Parallel_OK"])

summary_lines = []
summary_lines.append("=== Verification Summary Report ===\n")
summary_lines.append(f"Total number of panels in dataset       : {num_total}")
summary_lines.append(f"Number of panels used in best individual: {num_used}")
summary_lines.append(f"Number of panels not used               : {num_unused}\n")
summary_lines.append(f"Total number of strings                 : {len(string_df)}")
summary_lines.append(f"Strings failed voltage check (360–400V): {num_failed_voltage}")
summary_lines.append(f"Strings failed Imp check (±10%)        : {num_failed_imp}\n")
summary_lines.append(f"Total number of groups checked          : {len(group_df)}")
# === Global average over all strings ===
avg_string_voltage = string_df["Voltage(V)"].mean()
avg_string_imp_error = string_df["Imp Error (%)"].mean()

# === Add to summary report ===
summary_lines.append(f"\nAverage string voltage (V)              : {avg_string_voltage:.2f}")
summary_lines.append(f"Average string current error (%)        : {avg_string_imp_error:.2f}")

summary_lines.append("Overall Status: " +
                      (" PASS: All constraints satisfied." if (num_failed_voltage + num_failed_imp + num_failed_parallel) == 0
                       else " FAIL: Some constraints were violated."))

print("\n".join(summary_lines))

with open("panel_verification_summary.txt", "w") as f:
    f.write("\n".join(summary_lines))



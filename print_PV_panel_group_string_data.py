import os
import pandas as pd

def export_summary_to_txt(csv_path: str, filename: str = "summary.txt"):

    """
        Read the best_individual.csv file and export a summary of each string
        (within each group) to a TXT file in the same directory.
        """
    # Read data
    df = pd.read_csv(csv_path)

    # Group by Group and String index, aggregate panel count, total Vmp, Imp min/max
    summary = df.groupby(['Group', 'String']).agg(
        panels_count=('ID', 'count'),
        Vmp_sum=('Vmp', 'sum'),
        Imp_min=('Imp', 'min'),
        Imp_max=('Imp', 'max')
    ).reset_index()

    # Locate output directory (same as script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    txt_path = os.path.join(script_dir, filename)

    # Write TXT file
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("Panel Group Summary\n")
        f.write("===================\n\n")
        for row in summary.itertuples(index=False):
            group = int(row.Group)
            string = int(row.String)
            count = int(row.panels_count)
            v_sum = row.Vmp_sum
            i_min = row.Imp_min
            i_max = row.Imp_max
            power = v_sum * i_min

            f.write(f"Group {group}, String {string}:\n")
            f.write(f"  Number of panels: {count}\n")
            f.write(f"  Minimum Imp in string: {i_min:.2f} A\n")
            f.write(f"  Maximum Imp in string: {i_max:.2f} A\n")
            f.write(f"  Total Vmp (voltage): {v_sum:.2f} V\n")
            f.write(f"  Power (Vmp × Imp_min): {power:.2f} W\n")
            f.write("\n")

    print(f"Summary TXT report saved to {txt_path}")

if __name__ == "__main__":
    # 假设 best_individual.csv 与本脚本同目录
    export_summary_to_txt("best_individual.csv")

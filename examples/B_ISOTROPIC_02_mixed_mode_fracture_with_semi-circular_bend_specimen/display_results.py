from pathlib import Path
import csv
import matplotlib.pyplot as plt

# Parameters
B = 5e-3

# Directories to read
set_ids = [i + 1 for i in range(5)]
set_dirs = [Path(f"results_set_{i}") for i in set_ids]


# Function to read CSV as a dictionary of lists
def read_csv_as_dict(file_path):
    with open(file_path, "r") as f:
        reader = csv.DictReader(f)
        data = {col: [] for col in reader.fieldnames}
        for row in reader:
            for col in reader.fieldnames:
                data[col].append(float(row[col]))
        return data


# Read the CSV files
set_dfs = []
for set_dir in set_dirs:
    set_dfs.append(read_csv_as_dict(set_dir / "results.csv"))

# Load the references
ref_expe_csv = Path("reference/expe_beta_30.csv")
ref_expe_df = read_csv_as_dict(ref_expe_csv)
ref_gmts_csv = Path("reference/gmts_beta_30.csv")
ref_gmts_df = read_csv_as_dict(ref_gmts_csv)

# Reset the origin of the crack path
for df in set_dfs:
    xc_1_origin = df["xc_1"][1]
    xc_2_origin = df["xc_2"][1]
    df["xc_1"] = [x - xc_1_origin for x in df["xc_1"]]
    df["xc_2"] = [y - xc_2_origin for y in df["xc_2"]]

xc_1_expe_origin = ref_expe_df["xc_1"][0]
xc_2_expe_origin = ref_expe_df["xc_2"][0]
ref_expe_df["xc_1"] = [x - xc_1_expe_origin for x in ref_expe_df["xc_1"]]
ref_expe_df["xc_2"] = [y - xc_2_expe_origin for y in ref_expe_df["xc_2"]]

xc_1_gmts_origin = ref_gmts_df["xc_1"][0]
xc_2_gmts_origin = ref_gmts_df["xc_2"][0]
ref_gmts_df["xc_1"] = [x - xc_1_gmts_origin for x in ref_gmts_df["xc_1"]]
ref_gmts_df["xc_2"] = [y - xc_2_gmts_origin for y in ref_gmts_df["xc_2"]]

# # Load factor evolution
# plt.figure()
# for id, df in zip(set_ids, set_dfs):
#     plt.plot(df["a"], df["lambda"], label=f"Set {id}")
# plt.xlabel("Crack length (m)")
# plt.ylabel("Load factor")
# plt.legend()
# plt.grid()
#
# # Force displacement curve
# plt.figure()
# for id, df in zip(set_ids, set_dfs):
#     plt.plot(
#         [-l * u for l, u in zip(df["lambda"], df["uimp_2"])],
#         [-l * f * B for l, f in zip(df["lambda"], df["fimp_2"])],
#         marker=".",
#         label=f"Set {id}",
#     )
# plt.xlabel("Top displacement $-u$ [m]")
# plt.ylabel("Top force $-F$ [N]")
# plt.legend()
# plt.grid()
#
# Crack path
plt.figure()
plt.plot(ref_expe_df["xc_1"], ref_expe_df["xc_2"], marker=".", label="Aya06 (EXPE)")
plt.plot(ref_gmts_df["xc_1"], ref_gmts_df["xc_2"], marker=".", label="Aya06 (GMTS)")
for id, df in zip(set_ids, set_dfs):
    plt.plot(df["xc_1"], df["xc_2"], marker=".", label=f"Set {id}")
plt.xlabel("$x$-coordinate (m)")
plt.ylabel("$y$-coordinate (m)")
plt.axis("equal")
plt.legend()
plt.grid()
plt.savefig("crack_path.svg")

# Show the figures
plt.show()

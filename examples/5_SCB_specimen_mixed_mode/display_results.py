from pathlib import Path

import polars as pl
import matplotlib.pyplot as plt

# Parameters
B = 5e-3

# Directories to read
set_ids = [i + 1 for i in range(5)]
set_dirs = [Path(f"set_{i}") for i in set_ids]

# Read the CSV files
set_dfs = []
for set_dir in set_dirs:
    set_dfs.append(pl.read_csv(set_dir / "results.csv"))

# Load the references
ref_expe_csv = Path("reference/expe_beta_30.csv")
ref_expe_df = pl.read_csv(ref_expe_csv)
ref_gmts_csv = Path("reference/gmts_beta_30.csv")
ref_gmts_df = pl.read_csv(ref_gmts_csv)

# Reset the origin of the crack path
set_dfs = [
    df.with_columns(
        (pl.col("xc_1") - df["xc_1"][1]).alias("xc_1"),
        (pl.col("xc_2") - df["xc_2"][1]).alias("xc_2"),
    )
    for df in set_dfs
]
ref_expe_df = ref_expe_df.with_columns(
    (pl.col("xc_1") - ref_expe_df["xc_1"][0]).alias("xc_1"),
    (pl.col("xc_2") - ref_expe_df["xc_2"][0]).alias("xc_2"),
)
ref_gmts_df = ref_gmts_df.with_columns(
    (pl.col("xc_1") - ref_gmts_df["xc_1"][0]).alias("xc_1"),
    (pl.col("xc_2") - ref_gmts_df["xc_2"][0]).alias("xc_2"),
)

# Load factor evolution
plt.figure()
for id, df in zip(set_ids, set_dfs):
    plt.plot(df["a"], df["lambda"], label=f"Set {id}")
plt.xlabel("Crack length (m)")
plt.ylabel("Load factor")
plt.legend()
plt.grid()

# Force displacement curve
plt.figure()
for id, df in zip(set_ids, set_dfs):
    plt.plot(
        -df["lambda"] * df["uimp_2"],
        -df["lambda"] * df["fimp_2"] * B,
        marker=".",
        label=f"Set {id}",
    )
plt.xlabel("Top displacement $-u$ [m]")
plt.ylabel("Top force $-F$ [N]")
plt.legend()
plt.grid()

# Crack path
plt.figure()
plt.plot(ref_expe_df["xc_1"], ref_expe_df["xc_2"], marker=".", label="Aya06 (EXPE)")
plt.plot(ref_gmts_df["xc_1"], ref_gmts_df["xc_2"], marker=".", label="Aya06 (GMTS)")
for id, df in zip(set_ids, set_dfs):
    plt.plot(df["xc_1"], df["xc_2"], marker=".", label=f"Set {id}")
plt.xlabel("$x$-coordinate (m)")
plt.ylabel("$y$-coordinate (m)")
plt.axis('equal')
plt.legend()
plt.grid()

# Show the figures
plt.show()

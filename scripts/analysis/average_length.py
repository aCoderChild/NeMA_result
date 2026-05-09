from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CSV_PATH = Path(
	"results/full_npo/RAIL CrossLingual Transfer - MAPO_Results_Run_01.csv"
)
OUTPUT_PATH = Path("visualisation/average_length_mapo.png")
MODEL_PREFIX = "mapo_enesru_llama3_8b_"


def main() -> None:
	df = pd.read_csv(CSV_PATH)
	df.columns = df.columns.str.strip()

	df["source"] = df["source"].astype(str).str.strip()
	df["model"] = df["model"].astype(str).str.strip().str.replace(MODEL_PREFIX, "", regex=False)

	source_order = list(pd.unique(df["source"]))
	model_order = list(pd.unique(df["model"]))

	pivot = (
		df.pivot_table(index="source", columns="model", values="avg_length", aggfunc="mean")
		.reindex(index=source_order, columns=model_order)
	)

	fig_width = max(8, 0.9 * len(model_order))
	fig_height = max(4, 0.7 * len(source_order))
	fig, ax = plt.subplots(figsize=(fig_width, fig_height))

	matrix = pivot.to_numpy(dtype=float)
	im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")

	ax.set_xticks(range(len(model_order)))
	ax.set_xticklabels(model_order, rotation=45, ha="right")
	ax.set_yticks(range(len(source_order)))
	ax.set_yticklabels(source_order)
	ax.set_xlabel("model")
	ax.set_ylabel("source")
	ax.set_title("Average length by source and model")

	for row_idx in range(matrix.shape[0]):
		for col_idx in range(matrix.shape[1]):
			value = matrix[row_idx, col_idx]
			if pd.notna(value):
				ax.text(col_idx, row_idx, f"{value:.0f}", ha="center", va="center", color="black", fontsize=8)

	fig.colorbar(im, ax=ax, label="avg_length")
	fig.tight_layout()

	OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
	plt.close(fig)
	print(f"Saved heatmap to {OUTPUT_PATH}")


if __name__ == "__main__":
	main()

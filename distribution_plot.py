import pandas as pd
import matplotlib.pyplot as plt
import sys
from nltk.stem import WordNetLemmatizer
import os


def distribution_plot(ground_truth, predictions, title):

    # Combine predictions and ground truth terms into a DataFrame
    data = pd.DataFrame({"term": predictions + ground_truth})
    # Initialize the lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Lemmatize the terms
    # Convert all terms to lowercase
    data["term"] = data["term"].str.lower()
    data["term"] = data["term"].apply(
        lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()])
    )

    # Calculate the maximum length of terms in data["term"]

    max_len = 10

    data["term_length"] = [
        len(x.split()) if len(x.split()) < max_len else max_len for x in data["term"]
    ]

    # Group data by term length and calculate counts
    data_grouped = data.groupby("term_length").agg(
        true_count=("term", "count"),  # Total number of terms for each length
        correct_predictions=(
            "term",
            lambda x: x.isin(ground_truth).sum(),
        ),  # Correctly predicted terms
    )

    # Calculate number of incorrectly predicted terms
    data_grouped["incorrect_predictions"] = (
        data_grouped["true_count"] - data_grouped["correct_predictions"]
    )
    print(data_grouped)

    # Create a figure with appropriate size and style
    plt.figure(figsize=(12, 7), dpi=150)  # Increased size, higher resolution
    plt.style.use("ggplot")  # Apply a visually appealing style

    # Define bar widths for separate bars with emphasis on true terms
    bar_width = 0.35  # Adjust width as desired
    true_term_offset = -bar_width / 2  # Offset for true terms
    correct_prediction_offset = 0  # No offset for correct predictions
    incorrect_prediction_offset = bar_width / 2  # Offset for incorrect predictions

    # Create separate bars for each category with customized colors and gridlines
    plt.bar(
        data_grouped.index + true_term_offset,
        data_grouped["true_count"],
        bar_width,
        label="True Terms",
        color="royalblue",
        alpha=0.7,
    )  # Semi-transparent
    plt.bar(
        data_grouped.index + correct_prediction_offset,
        data_grouped["correct_predictions"],
        bar_width,
        label="Correct Predictions",
        color="forestgreen",
        alpha=0.7,
    )
    plt.bar(
        data_grouped.index + incorrect_prediction_offset,
        data_grouped["incorrect_predictions"],
        bar_width,
        label="Incorrect Predictions",
        color="coral",
        alpha=0.7,
    )

    plt.grid(axis="y", linestyle="--", alpha=0.6)  # Add subtle gridlines

    # Customize labels for clarity with font size and weight
    plt.xlabel("Term Length (Number of Words)", fontsize=14)  # , fontweight='bold')
    plt.ylabel("Number of Terms", fontsize=14)  # , fontweight='bold')
    plt.xticks(
        data_grouped.index, rotation=0, fontsize=12
    )  # Rotate and size x-axis labels
    plt.title(title, fontsize=16)  # , fontweight='bold')

    # Add a legend with a descriptive title, clear labels, and improved position
    plt.legend(
        title="Category", loc="upper right"
    )  # , bbox_to_anchor=(1, 1), fontsize=12)  # Legend outside plot, adjusted position

    # Adjust spacing for better layout and add minor ticks for improved readability
    plt.tight_layout()
    plt.minorticks_on()
    os.makedirs("img/distribution_plots", exist_ok=True)
    plt.savefig(f"img/distribution_plots/{title}", dpi=300, bbox_inches="tight")
    # Display the plot
    plt.show()


if __name__ == "__main__":
    # Load ground truth and predictions from files
    preds_path = sys.argv[1]
    if "total" in preds_path:
        gold_path = (
            "data/total/annotations/unique_annotations_lists/test_unique_terms.tsv"
        )
        title = "Total Dataset Term Length Distribution"
    elif "human" in preds_path:
        gold_path = (
            "data/human/annotations/unique_annotations_lists/test_unique_terms.tsv"
        )
        title = "Human Dataset Term Length Distribution Total"
    elif "kb" in preds_path:
        gold_path = "data/kb/annotations/unique_annotations_lists/test_unique_terms.tsv"
        title = "KB Dataset Term Length Distribution Total"

    ground_truth = pd.read_csv(gold_path, sep="\t", header=None)[0].tolist()
    predictions = pd.read_csv(preds_path, sep="\t", header=None)[0].tolist()
    distribution_plot(ground_truth, predictions, title)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    try:
        df = pd.read_csv("benchmark_results.csv")
    except FileNotFoundError:
        print("benchmark_results.csv not found! Run benchmark.py first.")
        return

    if df.empty:
        print("benchmark_results.csv is empty!")
        return

    # Set up the plot style
    sns.set_theme(style="whitegrid")

    # 1. Throughput Plot (Bar Chart)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="NumRequests", y="Throughput", hue="Engine", palette=["#ff7f0e", "#1f77b4"])
    plt.title("Inference Throughput: vLLM vs HuggingFace", fontsize=16)
    plt.xlabel("Number of Concurrent Requests (Batch Size)", fontsize=12)
    plt.ylabel("Throughput (tokens / sec)", fontsize=12)
    plt.legend(title="Engine")
    plt.tight_layout()
    plt.savefig("throughput_comparison.png", dpi=300)
    print("Saved throughput_comparison.png")

    # 2. Total Time Plot (Line Chart)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="NumRequests", y="TotalTime", hue="Engine", marker="o", palette=["#ff7f0e", "#1f77b4"])
    plt.title("Total Processing Time: vLLM vs HuggingFace", fontsize=16)
    plt.xlabel("Number of Concurrent Requests (Batch Size)", fontsize=12)
    plt.ylabel("Total Processing Time (seconds)", fontsize=12)
    plt.legend(title="Engine")
    plt.tight_layout()
    plt.savefig("total_time_comparison.png", dpi=300)
    print("Saved total_time_comparison.png")

if __name__ == "__main__":
    main()

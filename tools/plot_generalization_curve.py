import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot_curve(task_name):
    """
    Reads the CSV results file for a task and plots the generalization curve.
    """
    input_file = f"generalization_results_{task_name}.csv"
    
    try:
        # Read the data from the CSV file
        data = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Results file not found at '{input_file}'")
        print("Please run the 'run_generalization_sweep.sh' script first.")
        return

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(data['size_percent'], data['success_rate'], marker='o', linestyle='-', color='b')
    
    # Add a horizontal line for the baseline (100% size) performance
    baseline_success = data[data['size_percent'] == 100]['success_rate'].iloc[0]
    plt.axhline(y=baseline_success, color='r', linestyle='--', label=f'50% Success: {baseline_success:.1f}%')
    
    # Formatting
    plt.title(f'Generalization to Object Size - Task: {task_name}', fontsize=16)
    plt.xlabel('Object Size (% of Original)', fontsize=12)
    plt.ylabel('Success Rate (%)', fontsize=12)
    plt.xticks(data['size_percent'])
    plt.ylim(0, 105) # Y-axis from 0 to 100%
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    
    # Save and show the plot
    output_file = f'generalization_curve_{task_name}.png'
    plt.savefig(output_file)
    print(f"Plot saved to '{output_file}'")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot generalization curve from CSV results.")
    parser.add_argument("--task", required=True, type=str, help="The name of the task to plot (e.g., Lift, Door).")
    args = parser.parse_args()
    
    plot_curve(args.task)

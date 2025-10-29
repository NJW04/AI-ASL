import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- Configuration ---
cnn_log_file = Path("train_log.csv") # CNN without augmentation
cnn_aug_log_file = Path("aug_train_log.csv") # CNN with augmentation

output_dir = Path("plots") # Directory to save the plots
f1_plot_filename = "cnn_aug_vs_noaug_f1_comparison_mpl.png"
acc_plot_filename = "cnn_aug_vs_noaug_accuracy_comparison_mpl.png"

# --- Apply a nicer style ---
plt.style.use('seaborn-v0_8-whitegrid')

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# --- Load Data ---
log_files = {
    "CNN": cnn_log_file,
    "CNN_Aug": cnn_aug_log_file,
}

all_dfs = []
best_epochs = {} # MODIFICATION: Dictionary to store best epoch info
try:
    for model_name, file_path in log_files.items():
        df = pd.read_csv(file_path)
        rename_dict = {'train_acc': 'acc_train', 'val_acc': 'acc_val'}
        if 'val_macro_f1' in df.columns:
             rename_dict['val_macro_f1'] = 'macro_f1_val'
        
        if 'train_loss' in df.columns:
             df = df.rename(columns=rename_dict)
             
        df['model'] = model_name
        all_dfs.append(df)
        print(f"Loaded {model_name} log: {len(df)} epochs")

        # --- MODIFICATION: Find and store best epoch ---
        if 'macro_f1_val' in df.columns:
            best_idx = df['macro_f1_val'].idxmax() # Index of the row with max validation F1
            best_epoch_info = df.loc[best_idx]
            best_epochs[model_name] = best_epoch_info # Store the whole row (epoch, scores, etc.)
            print(f"  Best epoch for {model_name}: {int(best_epoch_info['epoch'])} (Val F1: {best_epoch_info['macro_f1_val']:.4f})")
        # --- END MODIFICATION ---

    df_all = pd.concat(all_dfs, ignore_index=True)

except FileNotFoundError as e:
    print(f"Error loading CSV file: {e}")
    print("Please make sure both CNN log files exist in the same directory as this script.")
    exit()
except Exception as e:
    print(f"An error occurred during data loading: {e}")
    exit()

colors = {'CNN': 'tab:blue', 'CNN_Aug': 'tab:green'}
markers = {'train': 'o', 'val': '^'}
linestyles = {'train': '-', 'val': '--'}

# --- Plot 1: Macro F1-Score Comparison ---
plt.figure(figsize=(12, 7))

for model_name in df_all['model'].unique():
    df_model = df_all[df_all['model'] == model_name]
    color = colors.get(model_name, 'black')

    if 'macro_f1_val' in df_model.columns:
        # Plot the regular validation F1 line
        plt.plot(df_model['epoch'], df_model['macro_f1_val'],
                 label=f'{model_name} Val F1', color=color,
                 linestyle=linestyles['val'], marker=markers['val'],
                 markersize=4, linewidth=2)

        # --- MODIFICATION: Plot the best epoch marker ---
        if model_name in best_epochs:
            best_info = best_epochs[model_name]
            plt.plot(best_info['epoch'], best_info['macro_f1_val'],
                     marker='*', color='red', markersize=12, linestyle='None', # Red star, no line
                     label=f'{model_name} Best Epoch ({int(best_info["epoch"])})' if plt.gca().get_legend() is None else None) # Only add label once
        # --- END MODIFICATION ---

    else:
        print(f"Warning: 'macro_f1_val' column not found for model {model_name}. Skipping F1 plot.")

plt.title('CNN vs. CNN+Augmentation: Validation Macro F1-Score Over Training', fontsize=16)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Validation Macro F1-Score', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

if 'macro_f1_val' in df_all.columns:
    min_val_f1 = df_all['macro_f1_val'].min()
    plt.ylim(bottom=max(0.8, min_val_f1 - 0.05))
else:
    plt.ylim(bottom=0.8)

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

f1_save_path = output_dir / f1_plot_filename
plt.tight_layout()
plt.savefig(f1_save_path)
print(f"Saved F1 comparison plot to: {f1_save_path}")
plt.close()

# --- Plot 2: Accuracy Comparison ---
plt.figure(figsize=(12, 7))

for model_name in df_all['model'].unique():
    df_model = df_all[df_all['model'] == model_name]
    color = colors.get(model_name, 'black')

    if 'acc_train' in df_model.columns:
        plt.plot(df_model['epoch'], df_model['acc_train'],
                 label=f'{model_name} Train Acc', color=color,
                 linestyle=linestyles['train'], marker=markers['train'],
                 markersize=4, linewidth=2)
    else:
        print(f"Warning: 'acc_train' column not found for model {model_name}.")

    if 'acc_val' in df_model.columns:
        # Plot the regular validation Accuracy line
        plt.plot(df_model['epoch'], df_model['acc_val'],
                 label=f'{model_name} Val Acc', color=color,
                 linestyle=linestyles['val'], marker=markers['val'],
                 markersize=4, linewidth=2)

        # --- MODIFICATION: Plot the best epoch marker ---
        # Note: We use the epoch determined by *best F1*, but plot the Acc value at that epoch
        if model_name in best_epochs:
            best_info = best_epochs[model_name]
            # Find the accuracy value at the best F1 epoch
            acc_at_best_f1_epoch = df_model.loc[df_model['epoch'] == best_info['epoch'], 'acc_val'].iloc[0]
            plt.plot(best_info['epoch'], acc_at_best_f1_epoch,
                     marker='*', color='red', markersize=12, linestyle='None', # Red star, no line
                     label=f'{model_name} Best F1 Epoch ({int(best_info["epoch"])})' if plt.gca().get_legend() is None else None) # Only add label once
        # --- END MODIFICATION ---

    else:
         print(f"Warning: 'acc_val' column not found for model {model_name}.")

plt.title('CNN vs. CNN+Augmentation: Accuracy Over Training', fontsize=16)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

if 'acc_val' in df_all.columns:
    min_val_acc = df_all['acc_val'].min()
    plt.ylim(bottom=max(0.8, min_val_acc - 0.05))
else:
    plt.ylim(bottom=0.8)

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

acc_save_path = output_dir / acc_plot_filename
plt.tight_layout()
plt.savefig(acc_save_path)
print(f"Saved Accuracy comparison plot to: {acc_save_path}")
plt.close()


import pandas as pd
import matplotlib.pyplot as plt

try:
    # Load the datasets
    df_hog = pd.read_csv("hog_epoch_log.csv")
    df_cnn = pd.read_csv("train_log.csv")
    df_aug = pd.read_csv("aug_train_log.csv")

    # --- Prepare Data for Matplotlib ---
    
    # Extract HOG + LR data
    epochs_hog = df_hog['epoch']
    # Use the original column name 'macro_f1_val' from hog_epoch_log.csv
    f1_hog = df_hog['macro_f1_val'] 

    # Extract Normal CNN data
    epochs_cnn = df_cnn['epoch']
    f1_cnn = df_cnn['val_macro_f1']

    # Extract CNN + Augmentation data
    epochs_aug = df_aug['epoch']
    f1_aug = df_aug['val_macro_f1']

    # Find the overall minimum F1 score to set y-axis limit
    min_f1 = min(f1_hog.min(), f1_cnn.min(), f1_aug.min())
    
    # --- Create Matplotlib Plot ---
    
    # Set the figure size
    plt.figure(figsize=(10, 6))
    
    # Plot data for each model with markers and labels
    plt.plot(epochs_hog, f1_hog, marker='o', linestyle='-', label='HOG + LR')
    plt.plot(epochs_cnn, f1_cnn, marker='s', linestyle='-', label='Normal CNN')
    plt.plot(epochs_aug, f1_aug, marker='^', linestyle='-', label='CNN + Augmentation')
    
    # --- Customize Plot ---
    
    # Set title and axis labels
    plt.title('Model Comparison: Validation Macro F1 vs. Epoch', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation Macro F1-Score', fontsize=12)
    
    # Set y-axis limits to be consistent with the previous Altair chart
    plt.ylim(min_f1 * 0.98, 1.02) 
    
    # Add a legend to identify the models
    plt.legend(title='Model', fontsize=10)
    
    # Add gridlines for better readability
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Adjust layout to prevent labels from being cut off
    plt.tight_layout()
    
    # --- Save the Plot ---
    
    # Save the figure as a PNG file
    output_filename = 'validation_macro_f1_comparison_mpl.png'
    plt.savefig(output_filename)
    
    print(f"Matplotlib graph saved as '{output_filename}'")

except FileNotFoundError as e:
    print(f"Error: Could not find file {e}. Please ensure all CSV files are present.")
except KeyError as e:
    print(f"Error: Missing expected column {e}. Please check the CSV file headers.")
except Exception as e:
    print(f"An error occurred: {e}")
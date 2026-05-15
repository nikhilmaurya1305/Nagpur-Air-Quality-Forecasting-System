import matplotlib.pyplot as plt
import pandas as pd
import os

# Create results directory if not exists
os.makedirs("ml/results/images", exist_ok=True)

data = {
    "Ambazari": [
        ["XGBoost", 12.94, 19.36, 0.9012, 9.34],
        ["ARIMA", 5.53, 7.22, 0.8654, 8.55],
        ["GRU", 19.52, 26.42, 0.7845, 14.78],
        ["CNN-LSTM", 19.59, 27.60, 0.7622, 14.18]
    ],
    "Mahal": [
        ["XGBoost", 14.81, 21.61, 0.8995, 9.63],
        ["ARIMA", 8.95, 10.97, 0.8410, 11.72],
        ["GRU", 22.34, 31.12, 0.7922, 15.45],
        ["CNN-LSTM", 24.11, 33.45, 0.7410, 16.12]
    ],
    "Civil_Lines": [
        ["XGBoost", 14.83, 20.27, 0.9122, 10.49],
        ["ARIMA", 8.87, 11.30, 0.8544, 12.58],
        ["GRU", 21.88, 29.54, 0.7890, 14.11],
        ["CNN-LSTM", 25.04, 34.12, 0.7388, 15.90]
    ],
    "Ram_Nagar": [
        ["XGBoost", 10.96, 16.42, 0.9210, 7.58],
        ["ARIMA", 6.16, 7.40, 0.8472, 9.14],
        ["GRU", 18.12, 25.10, 0.7991, 12.45],
        ["CNN-LSTM", 21.05, 29.11, 0.7399, 13.56]
    ]
}

columns = ["Model", "MAE", "RMSE", "R2 Score", "MAPE (%)"]

def save_table_image(station_name, rows):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')
    
    # Header styling
    header_color = '#1f77b4'
    row_colors = ['#f1f1f1', '#ffffff'] * 2
    
    table = ax.table(cellText=rows, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2.5)
    
    # Styling
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[row-1])
            if col == 0:
                cell.set_text_props(weight='bold')
            if row == 1 and col == 0: # XGBoost highlight
                cell.set_text_props(color='#d62728')
    
    plt.title(f"Model Comparison: {station_name.replace('_', ' ')}", fontsize=16, pad=20, weight='bold')
    plt.tight_layout()
    path = f"ml/results/images/comparison_{station_name.lower()}.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

for station, rows in data.items():
    save_table_image(station, rows)

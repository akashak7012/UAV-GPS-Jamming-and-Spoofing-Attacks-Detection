import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_flowchart():
    fig, ax = plt.subplots(figsize=(10, 14))
    ax.axis('off')

    steps = [
        "UAV Telemetry",
        "GPS / IMU / Magnetometer / Barometer",
        "Data Synchronization",
        "Data Cleaning",
        "Feature Engineering",
        "Feature Selection / Sensor Ablation",
        "Temporal 70/30 Split",
        "Machine Learning Models",
        "Gradient Boosting Primary Model",
        "UAV Attack Classification",
        "Benign / GPS Jamming / GPS Spoofing",
        "Evaluation + Robustness Analysis",
        "Real-Time Detection Simulation"
    ]
    
    num_steps = len(steps)
    y_start = 0.95
    y_step = 0.075
    box_height = 0.04
    box_width = 0.6
    
    # Define colors for different phases
    def get_color(idx):
        if idx < 7: # Data processing
            return '#e1f5fe', '#0288d1' 
        elif idx < 9: # Modeling
            return '#fff3e0', '#f57c00'
        else: # Evaluation / Output
            return '#e8f5e9', '#388e3c'

    for i, text in enumerate(steps):
        y = y_start - i * y_step
        
        bg_color, edge_color = get_color(i)
        
        # Draw box
        rect = patches.FancyBboxPatch(
            (0.5 - box_width/2, y - box_height/2),
            box_width, box_height,
            boxstyle="round,pad=0.01",
            fc=bg_color, ec=edge_color, lw=2,
            transform=ax.transAxes
        )
        ax.add_patch(rect)
        
        # Add text
        ax.text(
            0.5, y, text,
            ha='center', va='center',
            fontsize=12, fontweight='bold',
            transform=ax.transAxes
        )
        
        # Draw arrow to next box
        if i < num_steps - 1:
            next_y = y_start - (i + 1) * y_step
            ax.arrow(
                0.5, y - box_height/2 - 0.002,
                0, next_y + box_height/2 - (y - box_height/2) + 0.004,
                head_width=0.02, head_length=0.01,
                fc='black', ec='black',
                transform=ax.transAxes,
                length_includes_head=True
            )

    plt.title("UAV GPS Attack Detection - System Architecture", fontsize=16, fontweight='bold', y=0.98)
    
    out_path = os.path.join("results", "figures", "system_architecture.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Generated architecture diagram at: {out_path}")

if __name__ == "__main__":
    draw_flowchart()

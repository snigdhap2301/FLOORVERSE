from flask import Flask, request, jsonify, send_from_directory, send_file
import os
import numpy as np
import torch
import pandas as pd
import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
import matplotlib.patches as patches
from model import RPlanDataset, ConditionalVAE, generate_floorplan, load_model

app = Flask(__name__, static_folder='.')

# Global variables
model = None
dataset = None
initialized = False

def initialize():
    global model, dataset, initialized
    
    if not initialized:
        try:
            # Load dataset for normalization
            csv_path = "floorplan_dataset_improved.csv"  # Update with your CSV path
            dataset = RPlanDataset(csv_path)
            
            # Load trained model
            model_path = "C:/Users/Lenovo/OneDrive/Desktop/FLOOVERSE_FULL1/model_output/vae_model.pt"  # Update with your model path
            model = load_model(model_path)
            
            initialized = True
            print("Model and dataset initialized successfully")
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            import traceback
            traceback.print_exc()

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

@app.route('/generate_floorplan', methods=['POST'])
def generate_floorplan_endpoint():
    try:
        # Initialize if not already done
        if not initialized:
            initialize()
        
        if model is None or dataset is None:
            return jsonify({'error': 'Model or dataset not initialized'}), 500
        
        # Get data from request
        data = request.json
        
        # Extract parameters
        house_index = int(data.get('house_index', 0))
        unit = data.get('unit', 'metric')
        total_area = float(data.get('total_area', 100))
        rooms_data = data.get('rooms', [])
        
        # Count rooms by type
        room_counts = {}
        for room in rooms_data:
            room_type = room['type']
            count = room['count']
            room_counts[room_type] = count
        
        # Prepare parameters for the model
        params = {
            'num_rooms': sum(room_counts.values()),
            'bedrooms': room_counts.get('Bedroom', 1),
            'bathrooms': room_counts.get('Bathroom', 1),
            'kitchen': 1 if room_counts.get('Kitchen', 0) > 0 else 0,
            'living_room': 1 if room_counts.get('Living Room', 0) > 0 else 0,
            'dining_room': 1 if room_counts.get('Dining Room', 0) > 0 else 0,
            'study_room': 1 if room_counts.get('Office', 0) > 0 else 0,
            'balcony': 0,  # Default value
            'storage_room': 1 if room_counts.get('Storage Room', 0) > 0 else 0,
            'corridors': 1,  # Default value
            'total_area_sqm': total_area,
            'wall_thickness_cm': 15,  # Default value
            'num_doors': int(sum(room_counts.values()) * 1.5),  # Estimate based on rooms
            'num_windows': sum(room_counts.values()),  # Estimate based on rooms
            'floor_level': 1,  # Default value
            'avg_room_distance_m': 3  # Default value
        }
        
        # Generate floor plan
        floor_plan, efficiency = generate_floorplan(model, params, dataset)
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot floor plan
        ax1.imshow(floor_plan, cmap='gray')
        ax1.set_title('Generated Floor Plan')
        ax1.axis('off')
        
        # Plot efficiency metrics
        metrics = list(efficiency.keys())
        values = [efficiency[m] for m in metrics]
        
        ax2.bar(metrics, values)
        ax2.set_title('Space Efficiency Metrics')
        ax2.set_xticklabels(metrics, rotation=45, ha='right')
        ax2.set_ylim(0, max(values) * 1.2)
        
        for i, v in enumerate(values):
            ax2.text(i, v + 0.01, f"{v:.2f}", ha='center')
        
        plt.tight_layout()
        
        # Save the figure to a bytes buffer
        buf = io.BytesIO()
        FigureCanvas(fig).print_png(buf)
        plt.close(fig)
        buf.seek(0)
        
        return send_file(buf, mimetype='image/png')
    
    except Exception as e:
        print(f"Error generating floor plan: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


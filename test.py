import os
import torch
from unet3d_model import UNet3D
from config import config
from scipy.io import loadmat, savemat
import numpy as np

def load_model(model_path):
    model = UNet3D(in_channels=config.input_channels, out_channels=config.output_channels)
    checkpoint = torch.load(model_path, map_location=config.device)
    # If checkpoint contains state_dict, load it; otherwise, load checkpoint directly
    model.load_state_dict(checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint)
    model.to(config.device)
    model.eval()
    return model

def z_normalize(data, epsilon=1e-8):
    return (data - np.mean(data)) / (np.std(data) + epsilon)

def predict(model, mr_path, cbct_path):
    # Load MR and CBCT data from .mat files (according to keys defined in config)
    mr_data = loadmat(mr_path)[config.mr_key]
    cbct_data = loadmat(cbct_path)[config.cbct_key]
    
    # Apply Z-score normalization
    mr_data = z_normalize(mr_data)
    cbct_data = z_normalize(cbct_data)
    
    # Stack into a [2, H, W, D] array and add batch dimension
    input_data = np.stack([cbct_data, mr_data], axis=0)
    input_tensor = torch.from_numpy(input_data).float().unsqueeze(0).to(config.device)
    
    with torch.no_grad():
        transformed_image, dvf = model(input_tensor)
    
    # Return results with batch dimension removed
    return transformed_image.squeeze(0).cpu().numpy(), dvf.squeeze(0).cpu().numpy()

def main():
    # Path to the model checkpoint; modify as needed
    model_path = "./checkpoints/best_model.pth"
    model = load_model(model_path)
    
    # Test data folder paths
    test_mr_dir = "./dataset/test/MR"
    test_cbct_dir = "./dataset/test/CBCT"
    output_dir = "./outputs/test"
    
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(test_cbct_dir):
        if filename.endswith('.mat'):
            cbct_path = os.path.join(test_cbct_dir, filename)
            # Replace "CT_" with "MR_" to get the corresponding MR filename
            mr_filename = filename.replace("CT_", "MR_")
            mr_path = os.path.join(test_mr_dir, mr_filename)
            
            if os.path.exists(mr_path):
                transformed_image, predicted_dvf = predict(model, mr_path, cbct_path)
                
                # Save the predicted DVF
                dvf_filename = f"predicted_dvf_{filename}"
                dvf_path = os.path.join(output_dir, dvf_filename)
                savemat(dvf_path, {config.dvf_key: predicted_dvf})
                
                # Save the transformed MR image
                image_filename = f"transformed_image_{filename}"
                image_path = os.path.join(output_dir, image_filename)
                savemat(image_path, {'transformed_image': transformed_image})
                
                print(f"Processed {filename}")
            else:
                print(f"Skipped {filename}: No corresponding MR file found")

if __name__ == "__main__":
    main()

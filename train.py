import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import numpy as np
from unet3d_model import UNet3D, SpatialTransformationLayer
from config import config
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import binary_fill_holes

def fill_holes(mask):
    """
    Perform hole filling on the input binary mask.
    The mask has the shape [B, 1, H, W, D], and the returned filled mask has the same shape.
    """
    mask_np = mask.cpu().numpy()
    filled_mask = np.zeros_like(mask_np)
    for i in range(mask_np.shape[0]):
        filled = binary_fill_holes(mask_np[i, 0].astype(bool))
        filled_mask[i, 0] = filled.astype(np.float32)
    return torch.from_numpy(filled_mask).to(mask.device)

def setup_logger(save_dir):
    log_file = os.path.join(save_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class CustomDataset(Dataset):
    def __init__(self, cbct_dir, mr_dir, dvf_dir, logger=None):
        self.cbct_dir = cbct_dir
        self.mr_dir = mr_dir
        self.dvf_dir = dvf_dir
        self.cbct_files = sorted([f for f in os.listdir(self.cbct_dir) if f.endswith('.mat')])
        self.logger = logger

    def __len__(self):
        return len(self.cbct_files)

    def __getitem__(self, idx):
        cbct_file = self.cbct_files[idx]
        case_id = cbct_file.replace("CT_", "").replace(".mat", "")
        
        mr_file = f"MR_{case_id}.mat"
        dvf_file = f"dvf_{case_id}.mat"
        
        # Get full paths
        cbct_path = os.path.join(self.cbct_dir, cbct_file)
        mr_path = os.path.join(self.mr_dir, mr_file)
        dvf_path = os.path.join(self.dvf_dir, dvf_file)
        
        try:
            cbct_data = loadmat(cbct_path)[config.cbct_key]
            mr_data = loadmat(mr_path)[config.mr_key]
            dvf_data = loadmat(dvf_path)[config.dvf_key]
            
            # Data normalization
            cbct_data = (cbct_data - np.mean(cbct_data)) / (np.std(cbct_data) + 1e-8)
            mr_data = (mr_data - np.mean(mr_data)) / (np.std(mr_data) + 1e-8)
            
            # Process DVF data
            dvf_data = np.nan_to_num(dvf_data, nan=0.0)
            
            # Stack input data
            input_data = np.stack([cbct_data, mr_data], axis=0)
            
            # Convert to tensors
            input_tensor = torch.from_numpy(input_data).float()
            dvf_tensor = torch.from_numpy(dvf_data).float()
            
            # Check for NaN values
            if torch.isnan(input_tensor).any() or torch.isnan(dvf_tensor).any():
                raise ValueError(f"NaN values found in tensors for file {cbct_file}")
                
            return input_tensor, dvf_tensor
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error loading file {cbct_file}: {str(e)}")
            raise

def combined_loss(pred_dvf, target_dvf, transformed_image, mr_image, target_image):
    mse_loss = F.mse_loss(pred_dvf, target_dvf)
    
    spatial_transform = SpatialTransformationLayer()
    target_dvf_perm = target_dvf.permute(0, 4, 1, 2, 3)
    target_transformed_image = spatial_transform(mr_image, target_dvf_perm)
    image_similarity_loss = F.mse_loss(transformed_image, target_transformed_image)
    
    total_loss = (config.mse_weight * mse_loss + 
                  config.image_similarity_weight * image_similarity_loss)
    
    return total_loss, mse_loss.item(), image_similarity_loss.item()

def train_epoch(model, train_loader, optimizer, device, logger):
    model.train()
    running_total_loss = 0
    running_mse_loss = 0
    running_image_loss = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    
    for data, target_dvfs in pbar:
        try:
            data, target_dvfs = data.to(device), target_dvfs.to(device)
            optimizer.zero_grad()
            
            transformed_image, pred_dvf = model(data)
            mr_image = data[:, 1:2, ...]
            target_image = data[:, 0:1, ...]
            
            batch_total_loss, batch_mse_loss, batch_image_loss = combined_loss(
                pred_dvf, target_dvfs, transformed_image, mr_image, target_image)
            
            batch_total_loss.backward()
            optimizer.step()
            
            running_total_loss += batch_total_loss.item()
            running_mse_loss += batch_mse_loss
            running_image_loss += batch_image_loss
            
            pbar.set_postfix({"loss": f"{batch_total_loss.item():.4f}"})
            
        except Exception as e:
            logger.error(f"Error in batch: {str(e)}")
            continue
    
    num_batches = len(train_loader)
    return (running_total_loss / num_batches,
            running_mse_loss / num_batches,
            running_image_loss / num_batches)

def validate(model, val_loader, device, logger):
    model.eval()
    running_total_loss = 0
    running_mse_loss = 0
    running_image_loss = 0
    
    pbar = tqdm(val_loader, desc="Validation", leave=False)
    
    with torch.no_grad():
        for data, target_dvfs in pbar:
            try:
                data, target_dvfs = data.to(device), target_dvfs.to(device)
                transformed_image, pred_dvf = model(data)
                
                mr_image = data[:, 1:2, ...]
                target_image = data[:, 0:1, ...]
                
                batch_total_loss, batch_mse_loss, batch_image_loss = combined_loss(
                    pred_dvf, target_dvfs, transformed_image, mr_image, target_image)
                
                running_total_loss += batch_total_loss.item()
                running_mse_loss += batch_mse_loss
                running_image_loss += batch_image_loss
                
                pbar.set_postfix({"loss": f"{batch_total_loss.item():.4f}"})
                
            except Exception as e:
                logger.error(f"Error in validation batch: {str(e)}")
                continue
    
    num_batches = len(val_loader)
    return (running_total_loss / num_batches,
            running_mse_loss / num_batches,
            running_image_loss / num_batches)

def visualize_results(model, train_loader, val_loader, device, epoch, save_dir):
    """Visualize training and validation results, including raw and filled CBCT masks and ground truth DVF"""
    import matplotlib.pyplot as plt
    from scipy.ndimage import binary_fill_holes
    import numpy as np

    def fill_holes(mask):
        """
        Perform hole filling on the input binary mask.
        The mask has the shape [B, 1, H, W, D], and the returned filled mask has the same shape.
        """
        mask_np = mask.cpu().numpy()
        filled_mask = np.zeros_like(mask_np)
        for i in range(mask_np.shape[0]):
            filled = binary_fill_holes(mask_np[i, 0].astype(bool))
            filled_mask[i, 0] = filled.astype(np.float32)
        return torch.from_numpy(filled_mask).to(mask.device)

    model.eval()
    with torch.no_grad():
        for dataset_type, loader in [("train", train_loader), ("val", val_loader)]:
            num_samples = min(3, len(loader))
            for sample_idx in range(num_samples):
                data, target_dvfs = next(iter(loader))
                data, target_dvfs = data.to(device), target_dvfs.to(device)
                
                # Model prediction results
                transformed_image, pred_dvf = model(data)
                slice_idx = data.shape[4] // 2
                
                # Original CBCT mask and filled mask
                cbct_image = data[:, 0:1, ...]
                initial_mask = (torch.abs(cbct_image) >= config.liver_mask_threshold).float()
                filled_mask = fill_holes(initial_mask)

                # Extract ground truth DVF, assuming target_dvfs is already aligned with the model's output
                # Predicted and ground truth DVFs are both shaped [B, H, W, D, 3]
                
                # Begin plotting (6 rows x 4 columns layout, 24 subplots in total)
                plt.figure(figsize=(24, 30))
                print(f"{dataset_type.capitalize()} sample {sample_idx+1} shapes:")
                print(f"Data shape: {data.shape}")
                print(f"Transformed image shape: {transformed_image.shape}")
                print(f"Pred DVF shape: {pred_dvf.shape}")
                print(f"Target DVF shape: {target_dvfs.shape}")
                
                gs = plt.GridSpec(6, 4, hspace=0.5, wspace=0.3)
                
                # First row: input images
                ax0 = plt.subplot(gs[0, 0])
                cbct_slice = data[0, 0, :, :, slice_idx].cpu().numpy()
                ax0.imshow(cbct_slice, cmap="gray")
                ax0.set_title("Input CBCT")
                ax0.axis("off")
                
                ax1 = plt.subplot(gs[0, 1])
                mr_slice = data[0, 1, :, :, slice_idx].cpu().numpy()
                ax1.imshow(mr_slice, cmap="gray")
                ax1.set_title("Input MR")
                ax1.axis("off")
                
                # Second row: predictions
                ax2 = plt.subplot(gs[1, 0])
                transformed_slice = transformed_image[0, 0, 0, :, :, slice_idx].cpu().numpy()
                ax2.imshow(transformed_slice, cmap="gray")
                ax2.set_title("Predicted Transformed MR")
                ax2.axis("off")
                
                ax3 = plt.subplot(gs[1, 1])
                dvf_mag = torch.sqrt(torch.sum((pred_dvf)**2, dim=-1))
                dvf_mag_slice = dvf_mag[0, :, :, slice_idx].cpu().numpy()
                im = ax3.imshow(dvf_mag_slice, cmap="jet")
                ax3.set_title("Predicted DVF Magnitude")
                plt.colorbar(im, ax=ax3)
                ax3.axis("off")
                
                # Third row: mask comparison (original vs. filled)
                ax4 = plt.subplot(gs[2, 0])
                im4 = ax4.imshow(initial_mask[0, 0, :, :, slice_idx].cpu().numpy(), cmap="gray")
                ax4.set_title("Initial Mask")
                plt.colorbar(im4, ax=ax4)
                ax4.axis("off")
                
                ax5 = plt.subplot(gs[2, 1])
                im5 = ax5.imshow(filled_mask[0, 0, :, :, slice_idx].cpu().numpy(), cmap="gray")
                ax5.set_title("Filled Mask")
                plt.colorbar(im5, ax=ax5)
                ax5.axis("off")
                
                # Fourth row: predicted DVF components and DVF error
                ax6 = plt.subplot(gs[3, 0])
                pred_dvf_x = pred_dvf[0, :, :, slice_idx, 0].cpu().numpy()
                im6 = ax6.imshow(pred_dvf_x, cmap="jet")
                ax6.set_title("Predicted DVF (X)")
                plt.colorbar(im6, ax=ax6)
                ax6.axis("off")
                
                ax7 = plt.subplot(gs[3, 1])
                pred_dvf_y = pred_dvf[0, :, :, slice_idx, 1].cpu().numpy()
                im7 = ax7.imshow(pred_dvf_y, cmap="jet")
                ax7.set_title("Predicted DVF (Y)")
                plt.colorbar(im7, ax=ax7)
                ax7.axis("off")
                
                ax8 = plt.subplot(gs[3, 2])
                pred_dvf_z = pred_dvf[0, :, :, slice_idx, 2].cpu().numpy()
                im8 = ax8.imshow(pred_dvf_z, cmap="jet")
                ax8.set_title("Predicted DVF (Z)")
                plt.colorbar(im8, ax=ax8)
                ax8.axis("off")
                
                ax9 = plt.subplot(gs[3, 3])
                dvf_error = torch.mean(((pred_dvf - target_dvfs))**2, dim=-1)
                dvf_error_slice = dvf_error[0, :, :, slice_idx].cpu().numpy()
                im9 = ax9.imshow(dvf_error_slice, cmap="jet")
                ax9.set_title("DVF Error")
                plt.colorbar(im9, ax=ax9)
                ax9.axis("off")
                
                # Fifth row: Ground Truth DVF components
                ax10 = plt.subplot(gs[4, 0])
                target_dvf_x = target_dvfs[0, :, :, slice_idx, 0].cpu().numpy()
                im10 = ax10.imshow(target_dvf_x, cmap="jet")
                ax10.set_title("GT DVF (X)")
                plt.colorbar(im10, ax=ax10)
                ax10.axis("off")
                
                ax11 = plt.subplot(gs[4, 1])
                target_dvf_y = target_dvfs[0, :, :, slice_idx, 1].cpu().numpy()
                im11 = ax11.imshow(target_dvf_y, cmap="jet")
                ax11.set_title("GT DVF (Y)")
                plt.colorbar(im11, ax=ax11)
                ax11.axis("off")
                
                ax12 = plt.subplot(gs[4, 2])
                target_dvf_z = target_dvfs[0, :, :, slice_idx, 2].cpu().numpy()
                im12 = ax12.imshow(target_dvf_z, cmap="jet")
                ax12.set_title("GT DVF (Z)")
                plt.colorbar(im12, ax=ax12)
                ax12.axis("off")
                
                # Sixth row: differences between prediction and GT
                ax13 = plt.subplot(gs[5, 0])
                diff_x = pred_dvf_x - target_dvf_x
                im13 = ax13.imshow(diff_x, cmap="jet")
                ax13.set_title("Diff DVF (X)")
                plt.colorbar(im13, ax=ax13)
                ax13.axis("off")
                
                ax14 = plt.subplot(gs[5, 1])
                diff_y = pred_dvf_y - target_dvf_y
                im14 = ax14.imshow(diff_y, cmap="jet")
                ax14.set_title("Diff DVF (Y)")
                plt.colorbar(im14, ax=ax14)
                ax14.axis("off")
                
                ax15 = plt.subplot(gs[5, 2])
                diff_z = pred_dvf_z - target_dvf_z
                im15 = ax15.imshow(diff_z, cmap="jet")
                ax15.set_title("Diff DVF (Z)")
                plt.colorbar(im15, ax=ax15)
                ax15.axis("off")
                
                # Last subplot left blank
                ax16 = plt.subplot(gs[5, 3])
                ax16.axis("off")
                
                plt.suptitle(f"{dataset_type.capitalize()} Results - Epoch {epoch+1} - Sample {sample_idx+1}", fontsize=20)
                save_name = f"results_epoch_{epoch+1}_{dataset_type}_sample_{sample_idx+1}.png"
                plt.savefig(os.path.join(save_dir, save_name), bbox_inches="tight", dpi=300)
                plt.close()

def save_checkpoint(model, epoch, save_path, is_best=False):
    checkpoint = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
    }
    if is_best:
        save_path = os.path.join(os.path.dirname(save_path), "best_model.pth")
    torch.save(checkpoint, save_path)

def main():
    os.makedirs(config.save_dir, exist_ok=True)
    logger = setup_logger(config.save_dir)
    
    vis_dir = os.path.join(config.save_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    logger.info("Starting training process...")
    logger.info(f"Using device: {config.device}")
    
    train_dataset = CustomDataset(config.cbct_dir, config.mr_dir, config.dvf_dir, logger)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    
    val_dataset = CustomDataset(config.val_cbct_dir, config.val_mr_dir, config.val_dvf_dir, logger)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    model = UNet3D(in_channels=config.input_channels, out_channels=config.output_channels).to(config.device)
    
    start_epoch = 0
    if config.load_model and os.path.exists(config.model_path):
        checkpoint = torch.load(config.model_path, map_location=config.device)
        model.load_state_dict(checkpoint["state_dict"])
        start_epoch = checkpoint["epoch"]
        logger.info(f"Loaded checkpoint from epoch {start_epoch}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    for epoch in range(start_epoch, config.total_epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch+1}/{config.total_epochs}")
        logger.info(f"{'='*50}")
        
        train_loss, train_mse, train_image = train_epoch(model, train_loader, optimizer, config.device, logger)
        val_loss, val_mse, val_image = validate(model, val_loader, config.device, logger)
        
        logger.info("\nResults:")
        logger.info(f"{'':^10}{'Total':^12}{'MSE':^12}{'Image':^12}")
        logger.info(f"{'-'*46}")
        logger.info(f"{'Train':^10}{train_loss:^12.4f}{train_mse:^12.4f}{train_image:^12.4f}")
        logger.info(f"{'Val':^10}{val_loss:^12.4f}{val_mse:^12.4f}{val_image:^12.4f}")
        
        if (epoch + 1) % config.vis_interval == 0:
            visualize_results(model, train_loader, val_loader, config.device, epoch, vis_dir)
            logger.info(f"\nSaved visualizations for epoch {epoch+1}")
        
        if (epoch + 1) % config.save_interval == 0:
            save_path = os.path.join(config.save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            save_checkpoint(model, epoch, save_path)
            logger.info(f"Saved checkpoint for epoch {epoch+1}")
    
    final_save_path = os.path.join(config.save_dir, "final_model.pth")
    save_checkpoint(model, config.total_epochs-1, final_save_path)
    logger.info("Training completed! Final model saved.")

if __name__ == "__main__":
    main()

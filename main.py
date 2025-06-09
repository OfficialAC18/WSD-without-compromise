import os
import wandb
import torch
import logging
import argparse
from tqdm import tqdm
from utils.utils import get_dataset
from data.disentangled import DisentangledDataset
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', type=str, required=True, help='Directory to model checkpoints \
                                                            (models are trained if directory is empty)')
parser.add_argument('--model_name', type=str, help='Name of the model', default='trained_model.pth')
parser.add_argument('--latent_dim', type=int, default=10, help='Dimension of the latent space')
parser.add_argument('--dataset', type=str, default='dSprites', help='Name of the dataset to use')
parser.add_argument('--pipeline_seed', type=int, default=42, help='Seed for the pipeline')
parser.add_argument('--eval_seed', type=int, default=42, help='Seed for running evaluation')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing model checkpoints')
parser.add_argument('--model', type=str, default='G_VAE', help='Model to use (ML_VAE, G_VAE)')
parser.add_argument('--k_observed', type=int, default=1, help='Number of observed factors')
parser.add_argument('--observed_idx', type=str, default='constant', help='Index of the observed factors')
parser.add_argument('--aggregate', type=str, default='argmax', help='Aggregation method for the VAE')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer')
parser.add_argument('--batch_size', type=int, default=52, help='Batch size for training')
parser.add_argument('--num_train_steps', type=int, default=100000, help='Number of training steps')


def main(args):
    if not os.path.exists(args.model_dir) or args.overwrite:
        os.makedirs(args.model_dir, exist_ok=True)
        logging.warning(f'Training Models, saving to {args.model_dir}')
    else:
        logging.warning(f'Loading models from {args.model_dir}')

    #Load the relevant sampler
    sampler = get_dataset(args.dataset, seed=args.pipeline_seed)

    #Load the dataset object
    dataset = DisentangledDataset(sampler, observed_idx=args.observed_idx,
                                k_observed=args.k_observed)
    
    #Load the dataloader
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=1)
    
    
    #Load the model
    if args.model == 'ML_VAE':
        if args.aggregate == 'argmax':
            from models.ML_VAE import MLVAEArgMax
            model = MLVAEArgMax(data_shape=dataset.sampler.data_shape,
                                latent_dim=args.latent_dim,
                                num_channels=dataset.sampler.data_shape[0],
                                labels=False)
        else:
            from models.ML_VAE import MLVAELabels
            model = MLVAELabels(data_shape=dataset.sampler.data_shape,
                                latent_dim=args.latent_dim,
                                num_channels=dataset.sampler.data_shape[0],
                                labels=True)
    elif args.model == 'G_VAE':
        if args.aggregate == 'argmax':
            from models.GVAE import GroupVAEArgMax
            model = GroupVAEArgMax(data_shape=dataset.sampler.data_shape,
                                   latent_dim=args.latent_dim,
                                   num_channels=dataset.sampler.data_shape[0],
                                   labels=False)
        else:
            from models.GVAE import GroupVAELabels
            model = GroupVAELabels(data_shape=dataset.sampler.data_shape,
                                   latent_dim=args.latent_dim,
                                   num_channels=dataset.sampler.data_shape[0],
                                   labels=True)



    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate if hasattr(args, 'learning_rate') else 1e-3,
                                 betas = (0.9, 0.999), eps=1e-8)
    
    # Initialize wandb for experiment tracking
    wandb.init(
        project="disentangled-representations",
        config={
            "model": args.model,
            "dataset": args.dataset,
            "latent_dim": args.latent_dim,
            "aggregate": args.aggregate,
            "observed_idx": args.observed_idx,
            "k_observed": args.k_observed,
            "learning_rate": args.learning_rate if hasattr(args, 'learning_rate') else 1e-3,
            "batch_size": args.batch_size,
            "training_steps": args.training_steps if hasattr(args, 'training_steps') else 10000
        }
    )
    
    # Set device with priority: CUDA > MPS (Apple Silicon) > CPU
    import platform
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif platform.system() == 'Darwin' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    model.to(device)
    logging.info(f'Using device: {device}')
    
    # Training loop based on number of steps
    num_training_steps = args.training_steps if hasattr(args, 'training_steps') else 10000
    model.train()
    
    step = 0
    epoch = 0
    
    # Initialize tqdm progress bar
    pbar = tqdm(total=num_training_steps, desc="Training", unit="step")
    
    while step < num_training_steps:
        for batch_idx, batch_data in enumerate(dataloader):
            if step >= num_training_steps:
                break
            
            # Move data to device
            if isinstance(batch_data, torch.Tensor):
                batch_data = batch_data.to(device)
            elif isinstance(batch_data, (list, tuple)):
                batch_data = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch_data]

            # Forward pass
            x1_recons, x2_recons, loss, neg_elbo = model(batch_data)
            elbo = -neg_elbo
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress bar with current metrics
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'ELBO': f'{elbo.item():.4f}',
                'Epoch': epoch
            })
            pbar.update(1)
            
            # Log batch metrics to wandb
            if step % 100 == 0:
                wandb.log({
                    "loss": loss.item(),
                    "elbo": elbo.item(),
                    "step": step,
                    "epoch": epoch
                })
            
            # Print detailed progress less frequently
            if step % 1000 == 0:
                logging.info(f'Step [{step}/{num_training_steps}], Epoch: {epoch}, Loss: {loss.item():.4f}, ELBO: {elbo.item():.4f}')
            
            step += 1
        
        epoch += 1
    
    # Close progress bar
    pbar.close()
    
    # Save the trained model
    model_path = os.path.join(args.model_dir, args.model_name)
    torch.save(model.state_dict(), model_path)
    logging.info(f'Model saved to {model_path}')
    
    # Save model to wandb
    wandb.save(model_path)
    wandb.finish()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)




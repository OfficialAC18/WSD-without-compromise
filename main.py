import os
import logging
import argparse
from utils.utils import get_dataset
from data.disentangled import DisentangledDataset
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', type=str, required=True, help='Directory to model checkpoints \
                                                            (models are trained if directory is empty)')
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
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
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
                                k_observed=args.k_observed,
                                return_latents=True if args.aggregate == 'argmax' else False)
    
    #Load the dataloader
    dataloader = DataLoader(dataset,
                            batch_size=64,
                            shuffle=True,
                            num_workers=1)
    
    
    
    #Load the model
    if args.model == 'ML_VAE':
        if args.aggregate == 'argmax':
            from models.ML_VAE import MLVAEArgMax
            model = MLVAEArgMax(data_shape=dataset.sampler.data_shape,
                                latent_dim=args.latent_dim,
                                num_channels=dataset.sampler.data_shape[0])
        else:
            from models.ML_VAE import MLVAELabels
            model = MLVAELabels(data_shape=dataset.sampler.data_shape,
                                latent_dim=args.latent_dim,
                                num_channels=dataset.sampler.data_shape[0],
                                labels=dataset.sampler) #This is a proxy, not correct, need to refactor
    elif args.model == 'G_VAE':
        if args.aggregate == 'argmax':
            from models.GVAE import GroupVAEArgMax
            model = GroupVAEArgMax(data_shape=dataset.sampler.data_shape,
                                   latent_dim=args.latent_dim,
                                   num_channels=dataset.sampler.data_shape[0])
        else:
            from models.GVAE import GroupVAELabels
            model = GroupVAELabels(data_shape=dataset.sampler.data_shape,
                                   latent_dim=args.latent_dim,
                                   num_channels=dataset.sampler.data_shape[0],
                                   labels=dataset.sampler) #This is a proxy, not correct, need to refactor


    


    return




if __name__ == '__main__':
    args = parser.parse_args()
    main(args)




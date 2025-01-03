import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
import torch
import torchvision
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

from models.VAE import VAE
SEED = 42
LATENT_DIM = 5
BATCH_SIZE = 64
EPOCHS = 1000
MODEL_PATH = 'trained_models/VAE_LD_5'


def ELBOLoss(x_recons, x, z_mean, z_logvar):
    """
    Calculate the Evidence Lower Bound (ELBO) Loss for the VAE
    Args:
        x_recons: torch.Tensor, reconstructed images
        x: torch.Tensor, original images
        z_mean: torch.Tensor, mean of the latent space
        z_logvar: torch.Tensor, log variance of the latent space
    Returns:
        torch.Tensor, ELBO
    """
    BCE = F.binary_cross_entropy(x_recons.reshape(x_recons.shape[0],-1),
                                 x.reshape(x.shape[0],-1),
                                 reduction='mean')

    #Calculate the KL divergence
    kl_loss = torch.mean(-0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp()))
    
    elbo = BCE + kl_loss

    return elbo



#Set seed for experiment
torch.manual_seed(SEED)

#Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

#Load the dataset and create the dataloaders
train_dataset = torchvision.datasets.ImageFolder('datasets/playing_cards_minimal/train',
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.PILToTensor(),
                                                    torchvision.transforms.ConvertImageDtype(torch.float32)]))

val_dataset = torchvision.datasets.ImageFolder('datasets/playing_cards_minimal/val',
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.PILToTensor(),
                                                    torchvision.transforms.ConvertImageDtype(torch.float32)]))

train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=BATCH_SIZE, 
                                               shuffle=True,
                                               drop_last=True,
                                               num_workers=0)

val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=BATCH_SIZE,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=0)

#Initialize the model
model = VAE(data_shape = (3, 831, 523),
            latent_dim = LATENT_DIM,
            num_channels = 3)
model = model.to(device)
model.train(True)

#Load weights if available
if os.path.exists(os.path.join(MODEL_PATH,'best_model.pth')):
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH,'best_model.pth'),
                                     weights_only=True))


#Define the optimizer
optimizer = torch.optim.Adam(model.parameters(),
                             lr=1e-4)

#Define the loss function
loss_fn = ELBOLoss
#Just to keep tabs
loss_fn_recon_sum = nn.MSELoss(reduction='sum')
loss_fn_recon_mean = nn.MSELoss(reduction='mean')

def train():
    #Training loop
    epoch_num = 1
    train_loop_num = 1
    val_loop_num = 1
    best_vloss = float('inf')
    for _ in tqdm(range(EPOCHS), f"Epoch {epoch_num}/{EPOCHS}"):
        running_elbo_loss = 0.0
        running_recon_mean_loss = 0.0
        running_recon_sum_loss = 0.0

        for i, data in tqdm(enumerate(train_dataloader), f"Training Batch"):
            imgs, _ = data
            imgs = imgs.to(device)

            #Zero the gradients
            optimizer.zero_grad()

            #Generate the output
            output, z_mean, z_logvar = model(imgs)

            #Calculate the loss
            loss = loss_fn(output, imgs, z_mean, z_logvar)
            
            #Calculat some metrics
            loss_recon_mean = loss_fn_recon_mean(output, imgs)
            loss_recon_sum = loss_fn_recon_sum(output, imgs)

            #Backpropagate
            loss.backward()

            #Optimize
            optimizer.step()

            running_elbo_loss += loss.item()
            running_recon_mean_loss += loss_recon_mean.item()
            running_recon_sum_loss += loss_recon_sum.item()

            #Log the metrics
            wandb.log({'ELBO (Train)': loss.item(),
                    'Reconstruction Loss Mean (Train)': loss_recon_mean.item(),
                    'Reconstruction Loss Sum (Train)': loss_recon_sum.item()})
            
            train_loop_num += 1
        
        #Average the loss
        running_elbo_loss /= (i+1)
        running_recon_mean_loss /= (i+1)
        running_recon_sum_loss /= (i+1)

        #Log the metrics
        wandb.log({'Avg ELBO (Train)': running_elbo_loss,
                    'Avg Reconstruction Loss Mean (Train)': running_recon_mean_loss,
                    'Avg Reconstruction Loss Sum (Train)': running_recon_sum_loss})

        
        #Validation
        model.eval()
        running_elbo_vloss = 0.0
        running_recon_mean_vloss = 0.0
        running_recon_sum_vloss = 0.0
        with torch.no_grad():
            for i, vdata in tqdm(enumerate(val_dataloader), f"Validation Batch"):
                vimgs, _ = vdata
                vimgs = vimgs.to(device)

                #Generate the output
                voutput, vz_mean, vz_logvar = model(vimgs)

                #Calculate the loss
                vloss = loss_fn(voutput, vimgs, vz_mean, vz_logvar)
                
                #Calculat some metrics
                vloss_recon_mean = loss_fn_recon_mean(voutput, vimgs)
                vloss_recon_sum = loss_fn_recon_sum(voutput, vimgs)

                running_elbo_vloss += vloss.item()
                running_recon_mean_vloss += vloss_recon_mean.item()
                running_recon_sum_vloss += vloss_recon_sum.item()

                #Log the metrics
                wandb.log({'ELBO (Val)': vloss.item(),
                        'Reconstruction Loss Mean (Val)': vloss_recon_mean.item(),
                        'Reconstruction Loss Sum (Val)': vloss_recon_sum.item()})
                
                val_loop_num += 1
            
            #Average the loss
            running_elbo_vloss /= (i+1)
            running_recon_mean_vloss /= (i+1)
            running_recon_sum_vloss /= (i+1)

            #Log the metrics
            wandb.log({'Avg ELBO (Val)': running_elbo_vloss,
                        'Avg Reconstruction Loss Mean (Val)': running_recon_mean_vloss,
                        'Avg Reconstruction Loss Sum (Val)': running_recon_sum_vloss})

            if running_elbo_vloss < best_vloss:
                best_vloss = running_elbo_vloss
                torch.save(model.state_dict(), os.path.join(MODEL_PATH,'best_model.pth'))

            epoch_num += 1

if __name__ == '__main__':
    wandb.init(
    project='VAE Latent Dimension Test',

    config={
        'seed': SEED,
        'latent_dim': LATENT_DIM,
        'batch_size': BATCH_SIZE,
        'learning_rate': 1e-4,
        'dataset': 'playing_cards_minimal',
        'epochs': EPOCHS,
        'architecture': 'VAE LD-5',
    }
)
    train()
    wandb.finish()

    







import os
from data.dsprites import Dsprites
from data.playing_cards_no_transform import PlayingCardsNoTransform

def get_dataset(dataset_name, seed=42, latent_factor_indices=None):
    DATA_DIR = 'datasets/'
    if dataset_name == 'dSprites':
        return Dsprites(os.path.join(DATA_DIR,'dSprites'), seed=seed, observed_latent_factor_indices=latent_factor_indices)
    elif dataset_name == 'playing_cards_no_transform':
        return PlayingCardsNoTransform(os.path.join(DATA_DIR,'playing_card_games/simple_playing_cards'), seed=seed, observed_latent_factor_indices=latent_factor_indices)
    else:
        raise ValueError(f'Dataset {dataset_name} not implemented yet')

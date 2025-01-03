import torch
import random
from sklearn import linear_model

def generate_sample(ground_truth_data,
                    encoder_function,
                    num_examples,
                    batch_size,
                    seed):
    """
    Generate samples for the BetaVAE metric.
    
    Args:
        ground_truth_data: DisentangledDataset, ground truth data
        encoder_function: torch.nn.Module, encoder model which takes in input
        and returns latent encodings of the input
        num_examples: int, number of examples to generate
    
    Returns:
        points: torch.Tensor, (num_examples, latent_dim)
        labels: torch.Tensor, (num_examples)
    """
    random.seed(seed)
    points = torch.zeros(num_examples, encoder_function.latent_dim)
    labels = torch.zeros(num_examples)
    for idx in range(num_examples):
        index = random.randint(0, ground_truth_data.num_factors)
        _, factors1 = ground_truth_data.sample_observations(batch_size,    
                                                        return_factors=True)
        _, factors2 = ground_truth_data.sample_observations(batch_size,
                                                        return_factors=True)
        #Set sampled coordinate to be identical across the two samples
        factors2[:,index] = factors1[:,index]

        #Sample Observations based on the factors
        observations1 = ground_truth_data.sample_observations_from_factors(factors1)
        observations2 = ground_truth_data.sample_observations_from_factors(factors2)

        #Encode the observations
        encoding1 = encoder_function(observations1.to(encoder_function.device))
        encoding2 = encoder_function(observations2.to(encoder_function.device))

        #Compute the difference in the encodings as the feature vector
        feature_vector = torch.mean(torch.abs(encoding1 - encoding2), dim=0)
        points[idx,:] = feature_vector
        labels[idx] = index
    
    return points, labels

        
def beta_vae(ground_truth_data,
             encoder_function,
             num_train,
             num_test,
             batch_size=32,
             seed=42):
    """
    Computes the BetaVAE diesntanglement metric.

    Args:
        ground_truth_data: DisentangledDataset, ground truth data
        encoder_function: torch.nn.Module, encoder model which takes in input
        and returns latent encodings of the input
        num_train: int, number of training samples
        num_test: int, number of testing samples
        batch_size: int, batch size for the dataloader
    
    Returns:
        scores: dict, train and test BetaVAE scores
    """

    #Generate the training and testing samples
    train_points, train_labels = generate_sample(ground_truth_data,
                                                encoder_function,
                                                num_train,
                                                batch_size,
                                                seed)
    
    test_points, test_labels = generate_sample(ground_truth_data,
                                              encoder_function,
                                              num_test,
                                              batch_size,
                                              seed)
    
    model = linear_model.LogisticRegression(random_state=seed)
    model.fit(train_points.detach().cpu().numpy(), train_labels.detach().cpu().numpy())
    train_acc = torch.mean(torch.from_numpy(model.predict(train_points.detach().cpu().numpy()) == train_labels.detach().cpu().numpy()).float())
    test_acc = torch.mean(torch.from_numpy(model.predict(test_points.detach().cpu().numpy()) == test_labels.detach().cpu().numpy()).float())

    scores = {}
    scores['train_acc'] = train_acc
    scores['test_acc'] = test_acc
    return scores

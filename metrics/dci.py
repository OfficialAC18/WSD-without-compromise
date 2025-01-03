import scipy
import torch
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier

def compute_dci(ground_truth_data,
                encoder_function,
                num_train,
                num_test,
                surrogate_model = 'gbt',
                batch_size = 32):
    """
    Compute disentanglement, completeness and informativeness scores (DCI).
    Args:
        random_state: int, random seed
        ground_truth_data: DisentangledDataset, ground truth data
        encoder_function: torch.nn.Module, encoder model which takes in input
        and returns latent encodings of the input
        num_train: int, number of training samples
        num_test: int, number of testing samples
        surrogate_model: str, surrogate model to use
        bath_size: int, batch size for the dataloader
    
    Returns:
        scores: dict, average disentanglement, completeness and informativeness scores
        for the train and test sets.
    """
    #Generate the train set
    num_batches = num_train//batch_size
    last_batch = num_train - num_batches*batch_size

    for idx in range(num_batches + last_batch):
        if idx == num_batches:
            inputs_batch_train, y_batch_train = ground_truth_data.sample_observations(last_batch,    
                                                         return_factors=True)
        else:
            inputs_batch_train, y_batch_train = ground_truth_data.sample_observations(batch_size,    
                                                         return_factors=True)
        if idx > 0:
            x_train = torch.cat((x_train, encoder_function(inputs_batch_train)))
            y_train = torch.cat((y_train, y_batch_train))
        else:
            x_train = encoder_function(inputs_batch_train)
            y_train = y_batch_train

    #Generate the test set
    num_batches = num_test//batch_size
    last_batch = num_test - num_batches*batch_size

    for idx in range(num_batches + last_batch):
        if idx == num_batches:
            inputs_batch_test, y_batch_test = ground_truth_data.sample_observations(last_batch,    
                                                         return_factors=True)
        else:
            inputs_batch_test, y_batch_test = ground_truth_data.sample_observations(batch_size,    
                                                         return_factors=True)
        if idx > 0:
            x_test = torch.cat((x_test, encoder_function(inputs_batch_test)))
            y_test = torch.cat((y_test, y_batch_test))
        else:
            x_test = encoder_function(inputs_batch_test)
            y_test = y_batch_test

    assert x_train.shape[0] == num_train
    assert y_train.shape[0] == num_train
    assert x_test.shape[0] == num_test
    assert y_test.shape[0] == num_test

    #Compute Informativeness
    y_shape = y_train.shape[1]
    x_shape = x_train.shape[1]
    importance_matrix = torch.zeros(x_shape, y_shape,
                                    dtype=torch.float64)
    
    train_loss = []
    test_loss = []
    for i in range(y_shape):
        if surrogate_model == 'gbt':
            model = GradientBoostingClassifier()
        elif surrogate_model == 'adaboost':
            model = AdaBoostClassifier(n_estimators=100)
        
        model.fit(x_train.detach().numpy(), y_train[:,i].detach().numpy())
        importance_matrix[:,i] = torch.tensor(model.feature_importances_)
        train_loss.append(torch.mean(model.predict(x_train.detach().numpy()) == y_train[:,i].detach().numpy()))
        test_loss.append(torch.mean(model.predict(x_test.detach().numpy()) == y_test[:,i].detach().numpy()))

    train_loss = torch.mean(train_loss)
    test_loss = torch.mean(test_loss)

    #Compute Disentanglement
    per_latent_entropy = 1 - scipy.stats.entropy(importance_matrix.detach().numpy().T + 1e-12,
                                                 base = y_shape)
    
    if torch.sum(importance_matrix) == 0:
        importance_matrix = torch.ones_like(importance_matrix)
    
    latent_importance = torch.sum(importance_matrix, dim=1)/torch.sum(importance_matrix)
    disentanglement = torch.sum(latent_importance*per_latent_entropy)

    #Compute Completeness
    per_factor_entropy = 1 - scipy.stats.entropy(importance_matrix.detach().numpy() + 1e-12,
                                                 base = x_shape) 
    
    factor_importance = torch.sum(importance_matrix, dim=0)/torch.sum(importance_matrix)
    completeness = torch.sum(factor_importance*per_factor_entropy)

    scores = {}
    scores['informative_train'] = train_loss
    scores['informative_test'] = test_loss
    scores['disentanglement'] = disentanglement
    scores['completeness'] = completeness
    return scores

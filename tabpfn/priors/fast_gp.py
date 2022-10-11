import time

import torch
from torch import nn
import gpytorch

from .utils import get_batch_to_dataloader
from tabpfn.utils import default_device


# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def get_model(x, y, hyperparameters):
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1.e-9))
    model = ExactGPModel(x, y, likelihood)
    model.likelihood.noise = torch.ones_like(model.likelihood.noise) * hyperparameters["noise"]
    model.covar_module.outputscale = torch.ones_like(model.covar_module.outputscale) * hyperparameters["outputscale"]
    model.covar_module.base_kernel.lengthscale = torch.ones_like(model.covar_module.base_kernel.lengthscale) * \
                                                 hyperparameters["lengthscale"]
    return model, likelihood


@torch.no_grad()
def get_batch(batch_size, seq_len, num_features, device=default_device, hyperparameters=None,
              equidistant_x=False, fix_x=None, **kwargs):
    if isinstance(hyperparameters, (tuple, list)):
        hyperparameters = {"noise": hyperparameters[0]
            , "outputscale": hyperparameters[1]
            , "lengthscale": hyperparameters[2]
            , "is_binary_classification": hyperparameters[3]
            # , "num_features_used": hyperparameters[4]
            , "normalize_by_used_features": hyperparameters[5]
            , "order_y": hyperparameters[6]
            , "sampling": hyperparameters[7]
                           }
    elif hyperparameters is None:
        hyperparameters = {"noise": .1, "outputscale": .1, "lengthscale": .1}

    if 'verbose' in hyperparameters and hyperparameters['verbose']:
        print({"noise": hyperparameters['noise'], "outputscale": hyperparameters['outputscale']
                  , "lengthscale": hyperparameters['lengthscale'], 'batch_size': batch_size, 'sampling': hyperparameters['sampling']})

    # hyperparameters = {k: hyperparameters[k]() if callable(hyperparameters[k]) else hyperparameters[k] for k in
    #      hyperparameters.keys()}
    assert not (equidistant_x and (fix_x is not None))

    with gpytorch.settings.fast_computations(*hyperparameters.get('fast_computations', (True, True, True))):
        if equidistant_x:
            assert num_features == 1
            x = torch.linspace(0, 1., seq_len).unsqueeze(0).repeat(batch_size, 1).unsqueeze(-1)
        elif fix_x is not None:
            assert fix_x.shape == (seq_len, num_features)
            x = fix_x.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        else:
            if hyperparameters.get('sampling','uniform') == 'uniform':
                x = torch.rand(batch_size, seq_len, num_features, device=device)
            else:
                x = torch.randn(batch_size, seq_len, num_features, device=device)
        model, likelihood = get_model(x, torch.Tensor(), hyperparameters)
        model.to(device)
        # trained_model = ExactGPModel(train_x, train_y, likelihood).cuda()
        # trained_model.eval()
        is_fitted = False
        while not is_fitted:
            try:
                with gpytorch.settings.prior_mode(True):
                    model, likelihood = get_model(x, torch.Tensor(), hyperparameters)
                    model.to(device)

                    d = model(x)
                    d = likelihood(d)
                    sample = d.sample().transpose(0, 1)
                    is_fitted = True
            except RuntimeError: # This can happen when torch.linalg.eigh fails. Restart with new init resolves this.
                print('GP Fitting unsuccessful, retrying.. ')
                print(x)
                print(hyperparameters)

    if bool(torch.any(torch.isnan(x)).detach().cpu().numpy()):
        print({"noise": hyperparameters['noise'], "outputscale": hyperparameters['outputscale']
                  , "lengthscale": hyperparameters['lengthscale'], 'batch_size': batch_size})

    # TODO: Multi output
    return x.transpose(0, 1), sample, sample  # x.shape = (T,B,H)

DataLoader = get_batch_to_dataloader(get_batch)
DataLoader.num_outputs = 1

def get_model_on_device(x,y,hyperparameters,device):
    model, likelihood = get_model(x, y, hyperparameters)
    model.to(device)
    return model, likelihood


@torch.no_grad()
def evaluate(x, y, y_non_noisy, use_mse=False, hyperparameters={}, get_model_on_device=get_model_on_device, device=default_device, step_size=1, start_pos=0):
    start_time = time.time()
    losses_after_t = [.0] if start_pos == 0 else []
    all_losses_after_t = []

    with gpytorch.settings.fast_computations(*hyperparameters.get('fast_computations',(True,True,True))), gpytorch.settings.fast_pred_var(False):
        for t in range(max(start_pos, 1), len(x), step_size):
            loss_sum = 0.
            model, likelihood = get_model_on_device(x[:t].transpose(0, 1), y[:t].transpose(0, 1), hyperparameters, device)


            model.eval()
            # print([t.shape for t in model.train_inputs])
            # print(x[:t].transpose(0,1).shape, x[t].unsqueeze(1).shape, y[:t].transpose(0,1).shape)
            f = model(x[t].unsqueeze(1))
            l = likelihood(f)
            means = l.mean.squeeze()
            varis = l.covariance_matrix.squeeze()
            # print(l.variance.squeeze(), l.mean.squeeze(), y[t])

            assert len(means.shape) == len(varis.shape) == 1
            assert len(means) == len(varis) == x.shape[1]

            if use_mse:
                c = nn.MSELoss(reduction='none')
                ls = c(means, y[t])
            else:
                ls = -l.log_prob(y[t].unsqueeze(1))

            losses_after_t.append(ls.mean())
            all_losses_after_t.append(ls.flatten())
        return torch.stack(all_losses_after_t).to('cpu'), torch.tensor(losses_after_t).to('cpu'), time.time() - start_time

if __name__ == '__main__':
    hps = (.1,.1,.1)
    for redo_idx in range(1):
        print(
            evaluate(*get_batch(1000, 10, hyperparameters=hps, num_features=10), use_mse=False, hyperparameters=hps))

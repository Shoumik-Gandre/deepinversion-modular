import torch


def l2_regularization_total_variance(images: torch.Tensor) -> torch.Tensor:
    assert len(images.shape) == 4, "The input tensor may not be an image, shape != 4"
    # COMPUTE total variation regularization loss
    diff1 = images[:, :, :, :-1] - images[:, :, :, 1:]
    diff2 = images[:, :, :-1, :] - images[:, :, 1:, :]
    diff3 = images[:, :, 1:, :-1] - images[:, :, :-1, 1:]
    diff4 = images[:, :, :-1, :-1] - images[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    return loss_var_l2


def l1_regularization_total_variance(images: torch.Tensor) -> torch.Tensor:
    assert len(images.shape) == 4, "The input tensor may not be an image, shape != 4"
    # COMPUTE total variation regularization loss
    diff1 = images[:, :, :, :-1] - images[:, :, :, 1:]
    diff2 = images[:, :, :-1, :] - images[:, :, 1:, :]
    diff3 = images[:, :, 1:, :-1] - images[:, :, :-1, 1:]
    diff4 = images[:, :, :-1, :-1] - images[:, :, 1:, 1:]

    loss_var_l1 = (
        (diff1.abs() / 255.0).mean()
        + (diff2.abs() / 255.0).mean()
        + (diff3.abs() / 255.0).mean()
        + (diff4.abs() / 255.0).mean()
    )
    loss_var_l1 = loss_var_l1 * 255.0
    return loss_var_l1


def l2_image_priors(images: torch.Tensor) -> torch.Tensor:
    return torch.norm(images.view(images.shape[0], -1), dim=1).mean()


def regularize_feature_map(
        feature_map: torch.Tensor, 
        batchnorm_running_mean: torch.Tensor, 
        batchnorm_running_var: torch.Tensor
    ) -> torch.Tensor:
    """Regularize the input to a BatchNorm2d Layer with 
    norm between the difference of statistics (mean, var) of the input feature maps 
    and batchnormalization running statistics (running_mean, running_var)"""

    assert len(feature_map.shape) == 4
    
    num_channels = feature_map.shape[1]
    mean = feature_map.mean(dim=(0, 2, 3))
    var = (
        feature_map
            .permute(1, 0, 2, 3)
            .contiguous()
            .view((num_channels, -1))
            .var(1, unbiased=False)
    )

    return torch.norm(batchnorm_running_var.detach() - var, 2) \
            + torch.norm(batchnorm_running_mean.detach() - mean, 2)

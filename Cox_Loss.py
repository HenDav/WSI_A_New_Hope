import torch

def Cox_loss(risk_function_results: torch.Tensor, targets: torch.Tensor, censored: torch.Tensor) -> torch.float32:
    """
    This function implements the negative log partial likelihood used a loss.
    :param risk_function_results: risk feature results which is the result of beta_T * x, where x is the tile feature
    vector and beta is the coefficient vector.
    :param targets: targets representing the continous survival time
    :return:
    """
    num_tiles = targets.shape[0]
    # outer loop running over all live patients in the minibatch:
    # Actually we're going over all patients in the outer loop but in the inner loop we'll run over all patients still living
    # at the time the patient in the outer loop lives)
    loss = 0
    for i in range(num_tiles):
        if censored[i]:
            continue
        inner_loop_sum = 0
        for j in range(num_tiles):
            #  I'll assume that i in included in the inner summation
            # skipping patients j that died before patient i:
            if targets[j] < targets[i]:
                continue

            inner_loop_sum += torch.exp(risk_function_results[j])

        loss += risk_function_results[i] - torch.log(inner_loop_sum)

    return -loss

#def Gil_Loss(risk_function_results: torch.Tensor, targets: torch.Tensor, censored: torch.Tensor) -> torch.float32:

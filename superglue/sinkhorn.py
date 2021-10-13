import torch


def log_sinkhorn_iterations(z, log_mu, log_nu, iterations: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iterations):
        u = log_mu - torch.logsumexp(z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(z + u.unsqueeze(2), dim=1)
    return z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iterations: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    ms, ns = scores.new_tensor(m), scores.new_tensor(n)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = -(ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iterations)
    z = z - norm  # multiply probabilities by M+N
    return z

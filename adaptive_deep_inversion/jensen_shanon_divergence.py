import torch
import torch.nn as nn


class JensenShanonDivergence(nn.Module):
    """Module that returns an object with a callable Jenson Shanon Divergence function"""

    def __init__(self):
        super(JensenShanonDivergence, self).__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Jenson Shanon Divergence

        Returns:
            torch.Tensor: Jenson shanon divergence
        """
        p = p.view(-1, p.size(-1))
        q = q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log()))
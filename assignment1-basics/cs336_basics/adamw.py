import torch
import math
from typing import Any


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> Any:
        with torch.enable_grad():
            loss = None if closure is None else closure()
        # self.state keys by the params, and has the value moments.
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                t = state.get("t", 1)  # Get iteration number from the state, or 1.

                # compute the gradient of the loss
                g = p.grad  # Get the gradient of loss with respect to p.

                # compute adjusted alpha for iteration t
                at = lr * math.sqrt(1 - betas[1] ** t) / (1 - betas[0] ** t)

                # apply weight decay
                p -= lr * weight_decay * p  # Update weight tensor in-place.

                # update the first moment estimate, 𝛽1𝑚 + (1 − 𝛽1)𝑔
                m = betas[0] * m + (1 - betas[0]) * g
                state["m"] = m

                # update the second moment estimate, 𝛽2𝑣 + (1 − 𝛽2)𝑔2
                v = betas[1] * v + (1 - betas[1]) * g * g
                state["v"] = v

                # increment t
                state["t"] = t + 1  # Increment iteration number.

                # apply moment-adjusted weight updates
                p -= at * m / (torch.sqrt(v) + eps)

        return loss

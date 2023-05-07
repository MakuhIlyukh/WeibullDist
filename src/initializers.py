import torch


def k_initialize(m, k_init, manual_parametrization=False, eps=None):
    with torch.no_grad():
        if isinstance(k_init, str) and k_init == "random":
            res = (torch.rand(
                m, dtype=torch.float64) + 1)
        elif callable(k_init):
            res = (k_init())
        else:
            raise ValueError("k_init must be callable or 'random'")
        
        if manual_parametrization:
            if eps is None:
                raise ValueError("This is manual parametrization. Please specify eps")
            else:
                return torch.sqrt(res - eps)
        else:
            return res
            

def lmd_initialize(m, lmd_init, manual_parametrization=False, eps=None):
    with torch.no_grad():
        if isinstance(lmd_init, str) and lmd_init == "random":
            res = (torch.rand(
                m, dtype=torch.float64) + 1)
        elif callable(lmd_init):
            res = (lmd_init())
        else:
            raise ValueError("lmd_init must be callable or 'random'")
        
        if manual_parametrization:
            if eps is None:
                raise ValueError("This is manual parametrization. Please specify eps")
            else:
                return torch.sqrt(res - eps)
        else:
            return res
        

def q_initialize(m, q_init, manual_parametrization=False, c=None):
    with torch.no_grad():
        if isinstance(q_init, str):
            if q_init == 'dirichlet':
                res = torch.distributions.Dirichlet(
                        concentration=torch.full(
                            (m,), 1.0, dtype=torch.float64)
                    ).sample()
            elif q_init == "1/m":
                res = torch.full(
                    (m,), 1/m, dtype=torch.float64)
            else:
                raise ValueError("q_init must be 'dirchlet', '1/k' or callable")
        elif callable(q_init):
            res = q_init()
        else:
            raise ValueError("q_init must be 'dirchlet', '1/k' or callable")
        
        if manual_parametrization:
            if c is None:
                raise ValueError("This is manual parametrization. Please specify eps")
            else:
                return torch.log(res) + c
        else:
            return res
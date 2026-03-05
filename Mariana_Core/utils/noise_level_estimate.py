import torch.nn as nn
import torch

def noise_level_estimate(data, output = 'mean'):
    # patch segmentation
    [b, _, _, _] = data.shape
    if output == 'mean':
        V = torch.zeros([1]).to(data.device)
    else:
        V = torch.zeros([b]).to(data.device)
    seg = nn.Unfold(kernel_size=(9, 9), dilation=1, padding=0, stride=1)

    for k in range(b):
        patches = seg(data[k, :, :, :].unsqueeze(0))
        # principal component analysis
        out = torch.cov(patches.squeeze(0))
        L_complex, _ = torch.linalg.eig(out)
        sv = torch.real(L_complex)
        sv, _ = torch.sort(sv, descending=True, dim=0)
        # noise estimate
        for i in range(int(81)):
            t = torch.mean(sv[i:81])
            f = int((80 + i) / 2)
            f1 = f - 1
            f2 = min(f + 1, 80)
            if (t <= sv[f1]) and (t >= sv[f2]):
                if output == 'mean':
                    V = V + t
                else:
                    V[k] = t

                break

    return V / b if output == 'mean' else V
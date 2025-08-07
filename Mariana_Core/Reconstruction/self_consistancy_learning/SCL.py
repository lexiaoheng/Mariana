import torch
import torch.optim as optim
from .func.utils import model_train
from .func.model import CAE


def SCL(data, missing_mask, mode='zs-scl'):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = torch.tensor(data).unsqueeze(0).unsqueeze(0).to(device)
    missing_mask = torch.tensor(missing_mask).unsqueeze(0).unsqueeze(0).to(device)
    max_epoch = 8000
    lr = 0.001
    model = CAE(1, 8).to(device)

    _, _, h, w = data.shape

    loss_func = torch.nn.MSELoss()
    optimizer = optim.Adam (model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(max_epoch / 4), gamma = 0.9)

    model.train()
    model, out, loss1, loss2, loss3 = model_train(data=data, model=model, optimizer=optimizer, scheduler=scheduler,  loss_func=loss_func, epoch=max_epoch,
                    mode=mode, missing_mask=missing_mask)

    model.eval()
    return out.detach().cpu().squeeze(0).squeeze(0).numpy()




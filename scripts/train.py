import torch, model
from torch.amp import GradScaler, autocast

def run_training_epoch(vae: model.VAE, optimizer: torch.optim.Optimizer, train_loader: torch.utils.data.DataLoader,
                       scheduler: torch.optim.lr_scheduler._LRScheduler, scaler: GradScaler, mse_weight=1, kld_weight=1, subset_size=0, log_interval=16):
    vae_device = next(vae.parameters()).device
    vae.train() 

    for i, (batch, _) in enumerate(train_loader):
        original = batch.to(vae_device)

        optimizer.zero_grad()
        with autocast('cuda'):
            reconstruction, _, z_mean, z_log_var = vae(original)
            loss = vae.loss(original, reconstruction, z_mean, z_log_var, mse_weight, kld_weight, scheduler.get_last_lr()[0])
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if i % log_interval == 0:
            total_imgs = len(train_loader.dataset) if subset_size == 0 else subset_size
            batch_loss = loss.item() / len(batch)
            current_lr = scheduler.get_last_lr()[0]
            print(f'Epoch: {epoch} [{str(i * len(batch)).rjust(len(str(total_imgs)), " ")} / {total_imgs}]', end='    ')
            print(f'Loss: {batch_loss:.6f}', end='    ')
            print(f'LR: {current_lr:.6f}')
        
        if subset_size > 0 and i * len(batch) >= subset_size:
            break
        
    scheduler.step()

if __name__ == '__main__':
    from config import *
    import util, model, dataset, torch

    train_loader = dataset.get_train_loader(INPUT_SIZE, DATASET_NAME, DATASET_DIR)

    try:
        vae = model.VAE()
        vae.load(MODEL_NAME, MODEL_DIR)
        print(f'\nLoaded VAE: {MODEL_NAME} at epoch {vae.current_epoch}\n')
    except:
        vae = model.VAE(INPUT_SIZE, LATENT_SIZE, CONV_CHANNELS, FFN_LAYERS)
        print(f'\nCreated new VAE: {MODEL_NAME}\n')
    
    device = util.get_device()
    vae = vae.to(device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP, gamma=SCHEDULER_GAMMA)
    scaler = GradScaler()
    start_epoch = vae.current_epoch + 1

    for epoch in range(start_epoch, EPOCHS+start_epoch):
        run_training_epoch(vae, optimizer, train_loader, scheduler, scaler, MSE_WEIGHT, KLD_WEIGHT, SUBSET_SIZE)

        if epoch < EPOCHS + start_epoch - 1: 
            print(f'\nEpoch {epoch} complete. Saving checkpoint for "{MODEL_NAME}"...\n')
            vae.save(MODEL_NAME, MODEL_DIR, epoch, cp=True)

    print(f'\nTraining {MODEL_NAME} complete. Saving model "{MODEL_NAME}"...\n')
    vae.save(MODEL_NAME, MODEL_DIR, epoch)


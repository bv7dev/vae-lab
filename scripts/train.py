import torch, model

def run_training_epoch(vae: model.VAE, optimizer: torch.optim.Optimizer, train_loader: torch.utils.data.DataLoader, log_interval=16):
    vae_device = next(vae.parameters()).device
    vae.train() 

    for i, (batch, _) in enumerate(train_loader):
        original = batch.to(vae_device)

        optimizer.zero_grad()
        reconstruction, _, z_mean, z_log_var = vae(original)

        loss = vae.loss(original, reconstruction, z_mean, z_log_var, 1, 0.0012) #*(1 + 3*i/len(train_loader)))
        loss.backward()
        optimizer.step()
        
        if i % log_interval == 0:
            print(f'Epoch: {epoch} [{i * len(batch)}/{len(train_loader.dataset)}]', end='\t')
            print(f'Loss: {loss.item() / len(batch):.6f}')


if __name__ == "__main__":
    from config import *
    import util, model, dataset

    train_loader = dataset.get_train_loader(INPUT_SHAPE, DATASET_NAME, DATASET_DIR, SUBSET_SIZE)

    try:
        vae = model.VAE()
        vae.load(MODEL_NAME, MODEL_DIR)
        print(f"\nLoaded VAE: {MODEL_NAME} at epoch {vae.current_epoch}\n")
    except:
        vae = model.VAE(INPUT_SHAPE, LATENT_SIZE, CONV_CHANNELS)
        print(f"\nCreated new VAE: {MODEL_NAME}\n")
    
    device = util.get_device()
    vae = vae.to(device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=LEARNING_RATE)
    start_epoch = vae.current_epoch + 1

    for epoch in range(start_epoch, EPOCHS+start_epoch):
        run_training_epoch(vae, optimizer, train_loader)

        if epoch < EPOCHS + start_epoch - 1: 
            checkpoint_name = f"{MODEL_NAME}_checkpoint_{epoch}"
            print(f'\nEpoch {epoch} complete. Saving checkpoint "{checkpoint_name}"...\n')
            vae.save(checkpoint_name, MODEL_DIR, epoch)

    print(f'\nTraining {MODEL_NAME} complete. Saving model "{MODEL_NAME}"...\n')
    vae.save(MODEL_NAME, MODEL_DIR, epoch)
    

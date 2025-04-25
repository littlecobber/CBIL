import torch
import torch.nn as nn
from models.basic_videovae import CNNVideoEncoder, CNNVideoDecoder
from utils.datasets import DataLoader, DistributedSampler, VideoDataset, get_transforms_video
import torch.optim as optim
from tqdm import tqdm
import torch.utils.checkpoint as checkpoint


class VideoVAE(nn.Module):
    def __init__(
            self,
            grad_ckpt=True,  
            ch=128,
            ch_mult=(1, 1, 2, 2, 4),
            dropout=0.0,
    ):
        super(VideoVAE, self).__init__()
        self.encoder = CNNVideoEncoder(
            ch=ch, ch_mult=ch_mult, num_res_blocks=2, dropout=dropout,
            img_channels=3, output_channels=vocab_width, using_sa=True, using_mid_sa=True,
            grad_ckpt=grad_ckpt,
        )
        self.decoder = CNNVideoDecoder(
            ch=ch, ch_mult=ch_mult, num_res_blocks=3, dropout=dropout,
            input_channels=vocab_width, using_sa=True, using_mid_sa=True,
            grad_ckpt=grad_ckpt,
        )

    def forward(self, x):
        z = checkpoint.checkpoint(self.encoder, x, use_reentrant=False)
        reconstructed = checkpoint.checkpoint(self.decoder, z, use_reentrant=False)
        return reconstructed


def train(vae, dataloader, optimizer, num_epochs=1000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae.to(device)
    scaler = torch.amp.GradScaler('cuda')

    # for name, param in vae.named_parameters():
    #     if not param.requires_grad:
    #         print(f"Parameter {name} does not require grad!")

    for epoch in range(num_epochs):
        vae.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)

        for batch in progress_bar:
            video = batch['video'].to(device, dtype=torch.float16)  # FP16 
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                reconstructed = vae(video)
                # print(reconstructed.requires_grad)
                loss = vae_loss(reconstructed, video)
                # print(loss, loss.requires_grad)

            scaler.scale(loss).backward()  
            scaler.step(optimizer)  
            scaler.update()  

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}')
        if (epoch + 1) % 100 == 0:  
            torch.save(vae.state_dict(), f'vae_epoch_{epoch+1}.pth')


def vae_loss(reconstructed, original):
    return nn.MSELoss()(reconstructed, original)  



if __name__ == '__main__':
    torch.set_float32_matmul_precision("high")  

    root = ''

    dataset = VideoDataset(
        video_dir=root,
        transform=get_transforms_video(),
        num_frames=16,
        frame_interval=3,
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=1,
        rank=0,
        shuffle=True,
        seed=1
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    vae = VideoVAE(grad_ckpt=True) 
    optimizer = optim.AdamW(vae.parameters(), lr=1e-4)  

    train(vae=vae, dataloader=loader, optimizer=optimizer, num_epochs=1000)

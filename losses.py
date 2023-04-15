import torch
from audio_to_mel import Audio2Mel

def total_loss(fmap_real, logits_fake, fmap_fake, input_wav, output_wav, sample_rate=24000):
    relu = torch.nn.ReLU()
    l1Loss = torch.nn.L1Loss(reduction='mean')
    l2Loss = torch.nn.MSELoss(reduction='mean')
    loss = torch.tensor([0.0], device='cuda', requires_grad=True)
    l_t = torch.tensor([0.0], device='cuda', requires_grad=True)
    l_f = torch.tensor([0.0], device='cuda', requires_grad=True)
    l_g = torch.tensor([0.0], device='cuda', requires_grad=True)
    l_feat = torch.tensor([0.0], device='cuda', requires_grad=True)

    #time domain loss
    l_t = l1Loss(input_wav, output_wav) 
    #frequency domain loss
    for i in range(5, 11):
        fft = Audio2Mel(win_length=2 ** i, hop_length=2 ** i // 4, n_mel_channels=64, sampling_rate=sample_rate)
        l_f = l1Loss(fft(input_wav), fft(output_wav)) + l2Loss(fft(input_wav), fft(output_wav))
    
    #generator loss and feat loss
    for tt1 in range(len(fmap_real)):
        l_g = l_g + torch.mean(relu(1 - logits_fake[tt1])) / len(logits_fake)
        for tt2 in range(len(fmap_real[tt1])):
            l_feat = l_feat + l1Loss(fmap_real[tt1][tt2].detach(), fmap_fake[tt1][tt2]) / torch.mean(torch.abs(fmap_real[tt1][tt2].detach()))

    loss = 3*l_g + 3*l_feat + (l_t / 10) + l_f
    return loss

def disc_loss(logits_real, logits_fake):
    cx = torch.nn.ReLU()
    lossd = torch.tensor([0.0], device='cuda', requires_grad=True)
    for tt1 in range(len(logits_real)):
        lossd = lossd + torch.mean(cx(1-logits_real[tt1])) + torch.mean(cx(1+logits_fake[tt1]))
    lossd = lossd / len(logits_real)
    return lossd

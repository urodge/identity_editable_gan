identity_editable_gan/README.md
# Identity Editable GAN

This project demonstrates a GAN-based pipeline for identity-preserving high-resolution face generation, guided by textual prompts (e.g., "smiling face").

## Highlights
- Custom UNet-based generator with residual layers
- Perceptual, adversarial, and CLIP-based loss combinations
- Latent space interpretable via PCA/t-SNE (planned)
- Feature visualization via TensorBoard (to add)

## Skills Demonstrated
1. CNN/UNet architecture design
2. GAN with high-res output
3. Custom loss functions
4. Latent space exploration (roadmap)
5. Activation visualizations (roadmap)

## Requirements
- PyTorch
- torchvision
- clip (OpenAI)

## Run
```bash
python train.py

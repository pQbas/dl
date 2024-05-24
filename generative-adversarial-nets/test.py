import torch
from einops import rearrange, reduce, repeat
from model import Generator
from utils import Plotter, tensor2image
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your promgram')
    parser.add_argument('weights', type=str, help='Generator weights saved post training')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    generator = torch.load(args.weights)
    generator.to(device)
    generator.eval()
    plotter = Plotter(3)

    with torch.no_grad():
        # sampling minibatch of m noise samples from noise prior z ~ p_g(z)
        noise = torch.normal(mean=0.0, std=0.5, size=(10, 1, 100)).to(device)
        fake_data = generator(noise) # z ~ pg(z)
        fake_data = rearrange(fake_data, 'b s (c1 c2) -> b s c1 c2', c2=28)
        fake_data = fake_data.cpu()

        # plotting the fake image
        plotter.plot([(0, tensor2image(fake_data[0]),'imshow'),
                      (1, tensor2image(fake_data[1]),'imshow'),
                      (2, tensor2image(fake_data[2]),'imshow')])
        plotter.stop()

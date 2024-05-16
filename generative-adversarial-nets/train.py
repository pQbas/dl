import torch
from einops import rearrange, reduce, repeat
from model import Generator, Discriminator
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn as nn
from tqdm import tqdm
from utils import *
import pylab as pl
import argparse

class ModelTrainer:
    def __init__(self, discriminator, generator, dataset, loss_fn, discriminatorOptimizer, generatorOptimizer, batch_size):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        self.discriminator = discriminator.to(device)
        self.generator = generator.to(device)
        self.discriminatorOptimizer = discriminatorOptimizer
        self.generatorOptimizer = generatorOptimizer
        
        self.dataset = dataset
        self.loss_fn = loss_fn
        self.batch_size = batch_size

        self.dataloader = DataLoader(self.dataset, batch_size= self.batch_size, shuffle=True)


    def train(self):
        
        total_loss = 0

        '''
        Minibatch stochastic gradient descent training of generative adversarial nets
        implemented from paper Generative Adversarial Nets
        '''
        # sampling minibatch of m real samples from data distribution x ~ p_data(x)    
        real_data, _ = next(iter(self.dataloader))
        real_data = real_data.to(self.device)
        real_data = rearrange(real_data, 'b s c1 c2 -> b s (c1 c2)')

        # sampling minibatch of m noise samples from noise prior z ~ p_g(z)
        noise = torch.normal(mean=0.0, std=1.0, size=(real_data.shape[0], 1, 100)).to(self.device)
        fake_data = self.generator(noise) # z ~ pg(z)

        # Update the discriminator by ascending its stochastic gradient:
        fake = self.discriminator(fake_data)
        real = self.discriminator(real_data)
        lossDisc = self.loss_fn(fake, torch.zeros_like(fake)) + self.loss_fn(real, torch.ones_like(real))
        self.discriminatorOptimizer.zero_grad()
        lossDisc.backward(retain_graph=True)
        self.discriminatorOptimizer.step()

        # sampling minibatch of m noise samples from noise prior z ~ p_g(z)
        noise = torch.normal(mean=0.0, std=1.0, size=(real_data.shape[0], 1, 100)).to(self.device)
        fake_data = self.generator(noise) # z ~ pg(z)
        # Update the generator by descending its stochastic gradient
        fake = self.discriminator(fake_data)
        lossGen = self.loss_fn(fake, torch.ones_like(fake))
        self.generatorOptimizer.zero_grad()
        lossGen.backward()
        self.generatorOptimizer.step()
        return {
                 'generator': lossGen.item(),
                 'discriminator': lossDisc.item()
                }

    def sampler(self):
        with torch.no_grad():
            # sampling minibatch of m noise samples from noise prior z ~ p_g(z)
            noise = torch.normal(mean=0.0, std=1.0, size=(1, 1, 100)).to(self.device)
            fake_data = self.generator(noise) # z ~ pg(z)
            fake_data = rearrange(fake_data, 'b s (c1 c2) -> b s c1 c2', c2=28)
            return fake_data.to('cpu')



if __name__ == '__main__':
   
    parser = argparse.ArgumentParser(description='Description of your promgram')
    parser.add_argument('--batch', type=int, default=64, help='Batch size used during training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs used during training')
    parser.add_argument('--show', type=bool, default=False, help='Show an animation during trainig of image generated')

    args = parser.parse_args()

    # Create model, dataset, loss function, and optimizer
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=image2tensor_transformation
    )

    generator = Generator()
    discriminator = Discriminator()
    generatorOptimizer = torch.optim.SGD(params=generator.parameters(),lr=args.lr)
    discriminatorOptimizer = torch.optim.SGD(params=discriminator.parameters(),lr=args.lr)
    lossFunction = nn.BCELoss()
    
    # Create ModelTrainer instance
    trainer = ModelTrainer(discriminator, 
                           generator, 
                           training_data, 
                           lossFunction,
                           discriminatorOptimizer, 
                           generatorOptimizer,
                           batch_size = args.batch)
    
    # Training
    lossEvolution = {
                     'discriminator': [],
                     'generator': [] 
                    }
    
    plt.ion()
    figure1 = plt.figure()
    figure2 = plt.figure()

    for i in tqdm(range(args.epochs)):
        for i in range(10):
            currentLoss = trainer.train()
            lossEvolution['discriminator'].append(currentLoss['discriminator'])
            lossEvolution['generator'].append(currentLoss['generator'])
            
        imageSample = trainer.sampler()
        
        if args.show: 
            plt.figure(figure1.number)
            plt.plot(lossEvolution['discriminator'])
            plt.plot(lossEvolution['generator'])
            plt.draw()

            plt.figure(figure2.number)
            plt.imshow(tensor2image(imageSample[0]))
            plt.draw()

            plt.pause(0.0001)

    if args.show:  
        print('Press key "q" to exit')
        plt.show(block=True)



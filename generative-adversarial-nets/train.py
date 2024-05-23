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
import os
import wandb

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

    def getGenerator(self):
        return self.generator

    def getDiscriminator(self):
        return self.discriminator


if __name__ == '__main__':
   
    parser = argparse.ArgumentParser(description='Description of your promgram')
    parser.add_argument('--batch', type=int, default=32, help='Batch size used during training')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs used during training')
    parser.add_argument('--show', type=bool, default=False, help='Show an animation during trainig of image generated')
    parser.add_argument('--save', type=bool, default=True, help='Flag to activate save weights option')
    parser.add_argument('--path', type=str, default='weights', help='Default folder were to save the model')
    parser.add_argument('--wandb', type=bool, default=False, help='Flag to use weights and biases')
    parser.add_argument('--nbatch', type=int, default=-1, help='Number of batches used on training')

    args = parser.parse_args()

    # Create model, dataset, loss function, and optimizer instances
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=image2tensor_transformation
    )

    generator = Generator()
    discriminator = Discriminator()
    generatorOptimizer = torch.optim.SGD(params=generator.parameters(), lr=args.lr, momentum=args.momentum)
    discriminatorOptimizer = torch.optim.SGD(params=discriminator.parameters(),lr=args.lr, momentum=args.momentum)
    lossFunction = nn.BCELoss()
    
    # Create ModelTrainer instance
    trainer = ModelTrainer(discriminator, 
                           generator, 
                           training_data, 
                           lossFunction,
                           discriminatorOptimizer, 
                           generatorOptimizer,
                           batch_size = args.batch)
    
    # Training discriminator and generator
    lossEvolution = {
                     'discriminator': [],
                     'generator': [] 
                    }
    
    plotter = Plotter(3)

    if args.nbatch <= 0:
        args.nbatch = len(training_data)//args.batch

    if args.wandb:
        wandb.init(project='Generative Adversarial Network Experiments',
                config={'learning_rate': args.lr, 'epochs': args.epochs, 'batch_size': args.batch, 'number_batches': args.nbatch})

    for epoch in range(args.epochs):
        
        discriminatorLoss, generatorLoss = [],[]

        for i in tqdm(range(args.nbatch)):
            currentLoss = trainer.train()
            
            discriminatorLoss.append(currentLoss['discriminator'])
            generatorLoss.append(currentLoss['generator'])

        if args.wandb:
            wandb.log({'discriminator_loss': np.mean(discriminatorLoss), 
                    'generator_loss': np.mean(generatorLoss)},
                        step = epoch
                    )
        if args.show:
            lossEvolution['discriminator'].append(np.mean(discriminatorLoss))
            lossEvolution['generator'].append(np.mean(generatorLoss))
            imageSample = trainer.sampler()
            plotter.plot([(0, lossEvolution['discriminator'], 'plot'),
                          (1, lossEvolution['generator'], 'plot'),
                          (2, tensor2image(imageSample[0]),'imshow')])

    if args.show: 
        plotter.stop()

    # Save weights of the model
    if not args.save: 
        exit()

    # If save flag then save weights
    create_folder_if_not_exists(args.path)
    torch.save(trainer.getGenerator(), os.path.join(args.path,'generator.pt'))
    torch.save(trainer.getDiscriminator(), os.path.join(args.path,'discriminator.pt'))
    print('Trained weights were saved')



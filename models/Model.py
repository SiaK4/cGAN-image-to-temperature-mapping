"""
This model is modified based on the original implementation by Pix2Pix paper:
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master
"""

import torch
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F


class Img_to_T(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1 + lambda_cls * Pressure_loss
        By default, we use LSGAN loss + Temperature prediction loss + Pressure prediction loss, and UNet with batchnorm as the generator.
        """
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')  # Make sure to set to either netG = 'unet_256' or 'resnet_9blocks'
        if is_train:
            parser.set_defaults(pool_size=0,gan_mode='lsgan')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the model class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'G_cls', 'D_real', 'D_fake', 'D_cls']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.c = input['c'].to(self.device)
        
    def forward(self):

        c = self.c
        c = c.view(c.size(0),c.size(1),1,1)
        c = c.repeat(1,1,self.real_A.shape[2],self.real_A.shape[3])
       
        self.real_A_c = torch.cat([self.real_A,c],dim=1) # If you want to use the pressure as the conditioning parameter
        #self.real_A_c = self.real_A #If you do not want to include the pressure as the conditioning parameter

        self.fake_B = self.netG(self.real_A_c) #G(A|C)
        
    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A,self.fake_B),1)
        pred_fake, pred_fake_cls = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Discriminator classification loss
        #self.c_binary = self.c
        #self.c_binary[self.c_binary>0] = 1
        #self.loss_D_cls = F.binary_cross_entropy_with_logits(pred_fake_cls,self.c_binary,reduction='mean')
        #p_abs = torch.tensor([2,3,4,5,6,7,8]).to(self.device)
        #self.c_abs = torch.sum(self.c*p_abs,dim=1)
        #self.loss_D_cls = torch.mean((self.c_abs-pred_fake_cls)**2)
        
        #self.loss_D_cls = torch.mean(torch.sum(torch.abs(self.c - pred_fake_cls),dim=1))
        
        #print("self.c_abs:",self.c_abs)
        #print("pred_fake_cls:",pred_fake_cls)
        # Real
        #real_AB = torch.cat((self.real_A_c, self.real_B), 1)
        real_AB = torch.cat((self.real_A, self.real_B),1)
        pred_real,pred_real_cls = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        self.loss_D_cls = torch.mean(torch.sum(torch.abs(self.c-pred_real_cls),dim=1))
        
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 + self.opt.lambda_D_cls*self.loss_D_cls # Pressure prediction loss can be removed here for comparison purposes.
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake,pred_fake_cls = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)
        
        # Third, Pressure classification loss
        #self.loss_G_cls = F.binary_cross_entropy_with_logits(pred_fake_cls,self.c_binary,reduction='mean')
        #self.loss_G_cls = torch.mean((self.c_abs-pred_fake_cls)**2)
        self.loss_G_cls = torch.mean(torch.sum(torch.abs(self.c-pred_fake_cls),dim=1))
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1*self.opt.lambda_L1 + self.loss_G_cls*self.opt.lambda_D_cls #Pressure prediction loss can be removed here for comparison purposes.
        self.loss_G.backward()

    def optimize_parameters(self): 
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # update G's weights

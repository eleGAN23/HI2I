import os
from os.path import join as ospj
import time
import datetime
from munch import Munch
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from core.model import build_model
from core.checkpoint import CheckpointIO
from core.data_loader import InputFetcher
import core.utils as utils
from metrics.eval import calculate_metrics



class Solver(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda:'+str(args.gpu_num) if torch.cuda.is_available() else 'cpu')


        self.nets, self.nets_ema = build_model(args)
        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        for name, module in self.nets_ema.items():
            setattr(self, name + '_ema', module)

        if args.mode == 'train':
            self.optims = Munch()
            for net in self.nets.keys():
                if net == 'fan':
                    continue
                self.optims[net] = torch.optim.Adam(
                    params=self.nets[net].parameters(),
                    lr=args.f_lr if net == 'mapping_network' else args.lr,
                    betas=[args.beta1, args.beta2],
                    weight_decay=args.weight_decay)

            self.ckptios = [
                CheckpointIO(ospj(args.checkpoint_dir, args.experiment_name+'_{:06d}_nets.ckpt'), data_parallel=True, device=self.device,**self.nets),
                CheckpointIO(ospj(args.checkpoint_dir, args.experiment_name+'_{:06d}_nets_ema.ckpt'), data_parallel=True,device=self.device, **self.nets_ema),
                CheckpointIO(ospj(args.checkpoint_dir, args.experiment_name+'_{:06d}_optims.ckpt'), device=self.device,**self.optims)
        ]
        else:
            self.ckptios = [CheckpointIO(ospj(args.checkpoint_dir, args.experiment_name+'_{:06d}_nets_ema.ckpt'), data_parallel=True, device=self.device,**self.nets_ema)]

        self.to(self.device)
        for name, network in self.named_children():
            # Do not initialize the FAN parameters
            if ('ema' not in name) and ('fan' not in name):
                print('Initializing %s...' % name)
                network.apply(utils.he_init)

    def _save_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.save(step)

    def _load_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)
            print("loaded!")
 
        print("loaded checkpoints\n")

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def train(self, loaders):
        args = self.args
        nets = self.nets
        nets_ema = self.nets_ema
        optims = self.optims

        # fetch random validation images for debugging
        l_dim = args.latent_dim
        if args.phm and args.N == 3:
            l_dim = args.latent_dim-1
        fetcher = InputFetcher(loaders.src, loaders.ref, l_dim, 'train')
        fetcher_val = InputFetcher(loaders.val, None, args.latent_dim, 'val')
        inputs_val = next(fetcher_val)

        # resume training if necessary
        if args.resume_iter > 0:
            self._load_checkpoint(args.resume_iter)

        # remember the initial value of ds weight
        initial_lambda_ds = args.lambda_ds
        with wandb.init(config=args,project="quatstargan") as run:

            print('Start training...')
            start_time = time.time()
            for i in tqdm(range(args.resume_iter, args.total_iters),initial=args.resume_iter,total=args.total_iters,desc="training"):
                inputs = next(fetcher)
                x_real, y_org = inputs.x_src, inputs.y_src
                x_ref, x_ref2, y_trg = inputs.x_ref, inputs.x_ref2, inputs.y_ref
                z_trg, z_trg2 = inputs.z_trg, inputs.z_trg2


                x_real = x_real.to(self.device)
                x_ref = x_ref.to(self.device)
                x_ref2 = x_ref2.to(self.device)
                y_trg = y_trg.to(self.device)
                y_org = y_org.to(self.device)
                z_trg = z_trg.to(self.device)
                z_trg2 = z_trg2.to(self.device)

                if not args.real:
                    masks = nets.fan.get_heatmap(x_real[:,0:3,:,:]) if args.w_hpf > 0 else None
                else:
                    masks = nets.fan.get_heatmap(x_real) if args.w_hpf > 0 else None

                d_loss, d_losses_latent = compute_d_loss(nets, args, x_real, y_org, y_trg, z_trg=z_trg, masks=masks)
                self._reset_grad()
                d_loss.backward()
                optims.discriminator.step()
                
                d_loss, d_losses_ref = compute_d_loss(nets, args, x_real, y_org, y_trg, x_ref=x_ref, masks=masks)
                self._reset_grad()
        
                d_loss.backward()
                optims.discriminator.step()
        
                # train the generator
                g_loss, g_losses_latent = compute_g_loss(nets, args, x_real, y_org, y_trg, z_trgs=[z_trg, z_trg2], masks=masks)
                self._reset_grad()
                g_loss.backward()
        
                optims.generator.step()
                optims.mapping_network.step()
                optims.style_encoder.step()
        
                g_loss, g_losses_ref = compute_g_loss(nets, args, x_real, y_org, y_trg, x_refs=[x_ref, x_ref2], masks=masks)
                self._reset_grad()
                g_loss.backward()
                optims.generator.step()
                

                moving_average(nets.generator, nets_ema.generator, beta=0.999)
                moving_average(nets.mapping_network, nets_ema.mapping_network, beta=0.999)
                moving_average(nets.style_encoder, nets_ema.style_encoder, beta=0.999)
                if args.lambda_ds > 0:
                    args.lambda_ds -= (initial_lambda_ds / args.ds_iter)
        
                    # print out log info
                    if (i+1) % args.print_every == 0:
                        elapsed = time.time() - start_time
                        elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                        log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i+1, args.total_iters)
                        all_losses = dict()
                        for loss, prefix in zip([d_losses_latent, d_losses_ref, g_losses_latent, g_losses_ref],
                                                ['D/latent_', 'D/ref_', 'G/latent_', 'G/ref_']):
                            for key, value in loss.items():
                                all_losses[prefix + key] = value
                        all_losses['G/lambda_ds'] = args.lambda_ds
                        log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
                        wandb.log(all_losses,step=i+1,commit=True)
                        log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i+1, args.total_iters),
                        print( log,args.lambda_ds)
        
                    # generate images for debugging
                    if (i+1) % args.sample_every == 0:
                        os.makedirs(args.sample_dir, exist_ok=True)
                        utils.debug_image(nets_ema, args, inputs=inputs_val, step=i+1)

        
                    # save model checkpoints
                    if (i+1) % args.save_every == 0:
                        self._save_checkpoint(step=i+1)
        
                    # compute FID and LPIPS if necessary
                    if (i+1) % args.eval_every == 0 or (i+1)==10000:
                        lpip_lat, fid_lat = calculate_metrics(nets_ema, args, i+1, mode='latent')
                        wandb.log(dict(lpip_lat),step=i+2,commit=False)
                        wandb.log(dict(fid_lat),step=i+2,commit=False)

                        #ref
                        lpip_ref, fid_ref = calculate_metrics(nets_ema, args, i+1, mode='reference')
                        wandb.log(dict(lpip_ref),step=i+2,commit=False)
                        wandb.log(dict(fid_ref),step=i+2,commit=True)    
        
            run.finish()

    @torch.no_grad()
    def sample(self, loaders):
        args = self.args
        nets_ema = self.nets_ema
        self._load_checkpoint(args.resume_iter)
        fetcher_src = InputFetcher(loaders.src, None, args.latent_dim, 'test')
        fetcher_ref = InputFetcher(loaders.ref, None, args.latent_dim, 'test')
        for i in range(3):
            #if swithced bad news
            src = next(fetcher_src)
            ref = next(fetcher_ref)
            fname = ospj(args.result_dir, '{}_reference.jpg'.format(str(i)))
            print('Working on {}...'.format(fname))
            utils.translate_using_reference(nets_ema, args, src.x, ref.x, ref.y, fname)

            # fname = ospj(args.result_dir, '{}_video_ref.mp4'.format(str(i)))
            # print('Working on {}...'.format(fname))
            #video_ref(nets_ema, args, src.x, ref.x, ref.y, fname)
        # latent-guided image synthesis
        N =  src.x.size(0)
        y_trg_list = [torch.tensor(y).repeat(N).to(self.device) for y in range(min(args.num_domains, 5))]
        l_dim = args.latent_dim
        if args.phm and args.N == 3:
            l_dim = args.latent_dim-1
        z_trg_list = torch.randn(args.num_outs_per_domain, 1, l_dim).repeat(1, N, 1).to(self.device)
        for psi in [0.5, 0.7, 1.0]:
            filename = ospj(args.result_dir, args.experiment_name+'%06d_latent_psi_%.1f.jpg' % (i, psi))
            print('Working on {}...'.format(filename))

            utils.translate_using_latent(nets_ema, args, src.x, y_trg_list, z_trg_list, psi, filename)        

    @torch.no_grad()
    def evaluate(self,step):
        args = self.args
        nets_ema = self.nets_ema
        resume_iter = args.resume_iter
        self._load_checkpoint(args.resume_iter)
        lpip_lat, fid_lat = calculate_metrics(nets_ema, args, step, mode='latent')
        wandb.log(dict(lpip_lat),step=step,commit=False)
        wandb.log(dict(fid_lat),step=step,commit=False)
        #ref
        lpip_ref, fid_ref = calculate_metrics(nets_ema, args, step, mode='reference')
        wandb.log(dict(lpip_ref),step=step,commit=False)
        wandb.log(dict(fid_ref),step=step,commit=True)
        return lpip_lat,fid_lat,lpip_ref,fid_ref


def compute_d_loss(nets, args, x_real, y_org, y_trg, z_trg=None, x_ref=None, masks=None):
    assert (z_trg is None) != (x_ref is None)
    # with real images
    x_real.requires_grad_(True)
    out = nets.discriminator(x_real, y_org)
    loss_real = adv_loss(out, 1)
    loss_reg = r1_reg(out, x_real)
    # with fake images
    with torch.no_grad():
        if z_trg is not None:
            s_trg = nets.mapping_network(z_trg, y_trg)
        else:  # x_ref is not None
            s_trg = nets.style_encoder(x_ref, y_trg)
            #s_trg = torch.randn(8,64).to(self.device)
        x_fake = nets.generator(x_real, s_trg, masks=masks)
    out = nets.discriminator(x_fake, y_trg)
    loss_fake = adv_loss(out, 0)
    loss = loss_real + loss_fake + args.lambda_reg * loss_reg
    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item())


def compute_g_loss(nets, args, x_real, y_org, y_trg, z_trgs=None, x_refs=None, masks=None):
    assert (z_trgs is None) != (x_refs is None)
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
    if x_refs is not None:
        x_ref, x_ref2 = x_refs

    # adversarial loss
    if z_trgs is not None:
        s_trg = nets.mapping_network(z_trg, y_trg)
    else:
        s_trg = nets.style_encoder(x_ref, y_trg)

    x_fake = nets.generator(x_real, s_trg, masks=masks)
    #x_fake = torch.randn(8,4,128,128).to(self.device)
    
    out = nets.discriminator(x_fake, y_trg)
    loss_adv = adv_loss(out, 1)
    # style reconstruction loss
    s_pred = nets.style_encoder(x_fake, y_trg)
    #s_pred = torch.randn(8,64).to(self.device)
    
    #display(make_dot(nets.discriminator(x_fake, y_trg), params=dict(nets.discriminator.named_parameters()), show_attrs=True, show_saved=True))
    # print("pred",s_pred,"targ ",s_trg)
    # print("mean pred",torch.mean(s_pred),"MEAN TARG",torch.mean(s_trg))
    loss_sty = torch.mean(torch.abs(s_pred - s_trg))
    # diversity sensitive loss
    if z_trgs is not None:
        s_trg2 = nets.mapping_network(z_trg2, y_trg)
    else:
        s_trg2 = nets.style_encoder(x_ref2, y_trg)

    x_fake2 = nets.generator(x_real, s_trg2, masks=masks)

    
    x_fake2 = x_fake2.detach()
    
    #loss_ds = torch.mean(torch.abs((x_fake - x_fake2).torch()))
    loss_ds = torch.mean(torch.abs((x_fake - x_fake2)))
    # print("x_fake ",x_fake,"x fake2 ",x_fake2)
    # print("x fake",torch.mean(x_fake),"x fake2",torch.mean(x_fake2))
    # cycle-consistency loss
    if not args.real:
        masks = nets.fan.get_heatmap(x_fake[:,0:3,:,:]) if args.w_hpf > 0 else None
    else:
        masks = nets.fan.get_heatmap(x_fake) if args.w_hpf > 0 else None

    s_org = nets.style_encoder(x_real, y_org)
    
    x_rec = nets.generator(x_fake, s_org, masks=masks)
    
    #loss_cyc = torch.mean( torch.abs( (x_rec - x_real).torch() ) )
    loss_cyc = torch.mean( torch.abs( (x_rec - x_real) ) )

    loss = loss_adv + args.lambda_sty * loss_sty - args.lambda_ds * loss_ds + args.lambda_cyc * loss_cyc

    return loss, Munch(adv=loss_adv.item(),
                       sty=loss_sty.item(),
                       ds=loss_ds.item(),
                       cyc=loss_cyc.item())


def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)


def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg
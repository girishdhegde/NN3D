from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F


__author__ = "__Girish_Hegde__"


# def set_seed(seed):
#     # random.seed(seed)
#     # np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)


# def save_checkpoint(
#         net, optim, itr, val_loss, train_loss, best, filename, **kwargs,
#     ):
#     ckpt = {
#         'net':{
#             'config':net.get_config(),
#             'state_dict':net.state_dict(),
#         },
#         'optimizer':{
#             'state_dict':optim.state_dict(),
#         },
#         'training':{
#             'iteration':itr, 'val_loss':val_loss, 'train_loss':train_loss, 'best':best,
#         },
#         'kwargs':kwargs,
#     }
#     torch.save(ckpt, filename)
#     return ckpt


# def load_checkpoint(filename):
#     itr, best = 1, float('inf')
#     net_state, optim_state, kwargs = None, None, None
#     if filename is not None:
#         if Path(filename).is_file():
#             ckpt = torch.load(filename, map_location='cpu')
#             net_state = ckpt['net']['state_dict']
#             optim_state = ckpt['optimizer']['state_dict']
#             print('Model & Optim state dicts loaded successfully ...')
#             if 'training' in ckpt:
#                 itr, val_loss, train_loss, best = ckpt['training'].values()
#                 print('Training parameters loaded successfully ...')
#             if 'kwargs' in ckpt:
#                 kwargs = ckpt['kwargs']
#                 print('Additional kwargs loaded successfully ...')
#     return net_state, optim_state, itr, best, kwargs


# @torch.no_grad()
# def logits2text(logits, tokenizer, ):
#     """ Function to convert model prediction logits into words.
#         author: girish d. hegde
#     Args:
#         logits (torch.tensor[float]): [seq_len, vocab_size] - logits or [seq_len, ] - positions.
#         tokenizer (BPETokenizer): BPETokenizer.
#     Returns:
#         str: output string(set of words).
#     """
#     if logits.ndimension() == 2:
#         positions = torch.argmax(logits.detach().cpu(), dim=-1)
#     else:
#         positions = logits.detach().cpu().type(torch.int64)
#     text = tokenizer.decode(positions)
#     return text


# def write_pred(input_, logits, tokenizer, filename, label=''):
#     in_text = logits2text(input_, tokenizer)
#     out_text = logits2text(logits, tokenizer)
#     data = f'\n{"-"*100}\n{label}\n{"-"*100}\n<|INPUT|>\n{in_text}\n<|PREDICTION|>\n{out_text}\n{"-"*100}'
#     with open(filename,'a' if Path(filename).is_file() else 'w', encoding="utf-8") as fp:
#          fp.write(data)
#     return in_text, out_text


# @torch.no_grad()
# def sample(
#         prompt, net, tokenizer,
#         max_new_tokens=512, temperature=1.0, top_k=None, end_token=None,
#         device='cpu',
#     ):
#     """ Function to sample output text from trained model.
#         author: girish d. hegde
#     Args:
#         prompt (str): Any input sentence.
#         net (torch.nn.Module): trained model.
#         tokenizer (BPETokenizer): BPETokenizer tokenizer instance.
#         max_new_tokens (int): generate max_new_tokens.
#         temperature (float): controls randomness. 1 -> as it is(random), 0 -> precise.
#         top_k (int): top_k sampling. provides diversity.
#         end_token (int): end generation token.
#         device (torch.device): cpu or cuda.
#     Refs:
#         https://github.com/karpathy/nanoGPT/blob/master/model.py
#     Returns:
#         str: output string/text(prompt + prediction).
#         str: prediction.
#     """
#     net = net.to(device)

#     indices = tokenizer.encode(prompt)
#     indices = torch.tensor(indices, dtype=torch.int64, device=device)

#     output, prediction = net.generate(
#         indices, max_new_tokens, temperature, top_k, end_token
#     )

#     output = tokenizer.decode(output.to('cpu'))
#     prediction = tokenizer.decode(prediction.to('cpu'))
#     return output, prediction
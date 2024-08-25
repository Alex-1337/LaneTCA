import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .math import generate_permute_matrix

from .basic import seq_to_2d


class LSTNEngine(nn.Module):
    def __init__(self, lstn_model, gpu_id=0, long_term_mem_gap=9999, short_term_mem_skip=1):
        super().__init__()

        self.cfg = lstn_model.cfg
        self.align_corners = lstn_model.cfg.MODEL_ALIGN_CORNERS
        self.lstn = lstn_model

        self.max_obj_num = lstn_model.max_obj_num
        self.gpu_id = gpu_id
        self.long_term_mem_gap = long_term_mem_gap
        self.short_term_mem_skip = short_term_mem_skip
        self.losses = None

        self.restart_engine()




    def split_frames(self, xs, chunk_size):
        new_xs = []
        for x in xs:  
            all_x = list(torch.split(x, chunk_size, dim=0))
            new_xs.append(all_x)
        return list(zip(*new_xs))

    def add_reference_frame(self, img=None, mask=None, frame_step=-1, obj_nums=None, img_embs=None):
        if self.obj_nums is None and obj_nums is None:
            print('No objects for reference frame!')
            exit()
        elif obj_nums is not None:
            self.obj_nums = obj_nums

        if frame_step == -1:
            frame_step = self.frame_step

        # get enc_embs and mask_embs
        if img_embs is None:  
            curr_enc_embs, curr_one_hot_mask = self.encode_one_img_mask(img, mask, frame_step)
        else:
            _, curr_one_hot_mask = self.encode_one_img_mask(None, mask, frame_step)
            curr_enc_embs = img_embs

        if curr_enc_embs is None or curr_one_hot_mask is None:
            print('No image/mask for reference frame!')
            exit()

        if self.input_size_2d is None:
            self.update_size(img.size()[2:], curr_enc_embs[-1].size()[2:])

        self.curr_enc_embs = curr_enc_embs
        self.curr_one_hot_mask = curr_one_hot_mask

        # pos and id embedding
        if self.pos_emb is None:
            self.pos_emb = self.lstn.get_pos_emb(curr_enc_embs[-1])  \
                .expand(self.batch_size, -1, -1,-1).view(self.batch_size, -1, self.enc_hw).permute(2, 0, 1)  
 
        curr_id_emb = self.assign_identity(curr_one_hot_mask) 
        self.curr_id_embs = curr_id_emb  

        # self matching and propagation  
        self.curr_lsab_output = self.lstn.LSAB_forward(curr_enc_embs, None, None, curr_id_emb, pos_emb=self.pos_emb, size_2d=self.enc_size_2d)
        lsab_embs, lsab_curr_memories, lsab_long_memories, lsab_short_memories = self.curr_lsab_output

        if self.long_term_memories is None:
            self.long_term_memories = lsab_long_memories
        else:
            self.update_long_term_memory(lsab_long_memories)

        self.last_mem_step = self.frame_step
        self.short_term_memories_list = [lsab_short_memories]
        self.short_term_memories = lsab_short_memories


    def assign_identity(self, one_hot_mask):
        if self.enable_id_shuffle: 
            one_hot_mask = torch.einsum('bohw, bot->bthw', one_hot_mask, self.id_shuffle_matrix)
        id_emb = self.lstn.get_id_emb(one_hot_mask).view(self.batch_size, -1, self.enc_hw).permute(2, 0, 1) 

        if self.training and self.freeze_id:
            id_emb = id_emb.detach()

        return id_emb

    def match_propogate_one_frame(self, img=None, img_embs=None):
        self.frame_step += 1
        if img_embs is None:
            curr_enc_embs, _ = self.encode_one_img_mask(img, None, self.frame_step)           
        else:
            curr_enc_embs = img_embs
        self.curr_enc_embs = curr_enc_embs
        self.curr_lsab_output = self.lstn.LSAB_forward(curr_enc_embs, self.long_term_memories, self.short_term_memories, None,          
                                                      pos_emb=self.pos_emb, size_2d=self.enc_size_2d)










    def update_long_term_memory(self, new_long_term_memories):
        if self.long_term_memories is None:
            self.long_term_memories = new_long_term_memories
        updated_long_term_memories = []
        for new_long_term_memory, last_long_term_memory in zip(new_long_term_memories, self.long_term_memories):
            updated_e = []
            for new_e, last_e in zip(new_long_term_memory, last_long_term_memory):
                if new_e is None or last_e is None:
                    updated_e.append(None)
                else:
                    updated_e.append(torch.cat([new_e, last_e], dim=0))
            updated_long_term_memories.append(updated_e)
        self.long_term_memories = updated_long_term_memories


class LSTNInferEngine(nn.Module):
    def __init__(self, lstn_model, gpu_id=0, long_term_mem_gap=9999, short_term_mem_skip=1, max_lstn_obj_num=None):
        super().__init__()

        self.cfg = lstn_model.cfg
        self.lstn = lstn_model

        if max_lstn_obj_num is None or max_lstn_obj_num > lstn_model.max_obj_num:
            self.max_lstn_obj_num = lstn_model.max_obj_num
        else:
            self.max_lstn_obj_num = max_lstn_obj_num

        self.gpu_id = gpu_id
        self.long_term_mem_gap = long_term_mem_gap
        self.short_term_mem_skip = short_term_mem_skip

        self.lstn_engines = []

        self.restart_engine()

    def restart_engine(self):
        del (self.lstn_engines)
        self.lstn_engines = []
        self.obj_nums = None

    def separate_mask(self, mask, obj_nums):
        if mask is None:
            return [None] * len(self.lstn_engines)
        if len(self.lstn_engines) == 1:
            return [mask], [obj_nums]

        separated_obj_nums = [self.max_lstn_obj_num for _ in range(len(self.lstn_engines))]
        if obj_nums % self.max_lstn_obj_num > 0:
            separated_obj_nums[-1] = obj_nums % self.max_lstn_obj_num

        if len(mask.size()) == 3 or mask.size()[0] == 1:
            separated_masks = []
            for idx in range(len(self.lstn_engines)):
                start_id = idx * self.max_lstn_obj_num + 1
                end_id = (idx + 1) * self.max_lstn_obj_num
                fg_mask = ((mask >= start_id) & (mask <= end_id)).float()
                separated_mask = (fg_mask * mask - start_id + 1) * fg_mask
                separated_masks.append(separated_mask)
            return separated_masks, separated_obj_nums
        else:
            prob = mask
            separated_probs = []
            for idx in range(len(self.lstn_engines)):
                start_id = idx * self.max_lstn_obj_num + 1
                end_id = (idx + 1) * self.max_lstn_obj_num
                fg_prob = prob[start_id:(end_id + 1)]
                bg_prob = 1. - torch.sum(fg_prob, dim=1, keepdim=True)
                separated_probs.append(torch.cat([bg_prob, fg_prob], dim=1))
            return separated_probs, separated_obj_nums


    def soft_logit_aggregation(self, all_logits):
        if len(all_logits) == 1:
            return all_logits[0]

        fg_probs = []
        bg_probs = []

        for logit in all_logits:
            prob = torch.softmax(logit, dim=1)
            bg_probs.append(prob[:, 0:1])
            fg_probs.append(prob[:, 1:1 + self.max_lstn_obj_num])

        bg_prob = torch.prod(torch.cat(bg_probs, dim=1), dim=1, keepdim=True)
        merged_prob = torch.cat([bg_prob]+fg_probs, dim=1).clamp(1e-5, 1 - 1e-5)
        merged_logit = torch.logit(merged_prob)

        return merged_logit

    def add_reference_frame(self, img, mask, obj_nums, frame_step=-1):
        if isinstance(obj_nums, list):
            obj_nums = obj_nums[0]
        self.obj_nums = obj_nums
        lstn_num = max(np.ceil(obj_nums / self.max_lstn_obj_num), 1)
        while (lstn_num > len(self.lstn_engines)):
            new_engine = LSTNEngine(self.lstn, self.gpu_id, self.long_term_mem_gap, self.short_term_mem_skip)
            new_engine.eval()
            self.lstn_engines.append(new_engine)

        separated_masks, separated_obj_nums = self.separate_mask(mask, obj_nums)
        img_embs = None
        for lstn_engine, separated_mask, separated_obj_num in zip(self.lstn_engines, separated_masks, separated_obj_nums):
            lstn_engine.add_reference_frame(img, separated_mask, obj_nums=[separated_obj_num], frame_step=frame_step, img_embs=img_embs)
            if img_embs is None:  # reuse image embeddings
                img_embs = lstn_engine.curr_enc_embs

        self.update_size()

    def match_propogate_one_frame(self, img=None):
        img_embs = None
        for lstn_engine in self.lstn_engines:
            lstn_engine.match_propogate_one_frame(img, img_embs=img_embs)
            if img_embs is None:  # reuse image embeddings
                img_embs = lstn_engine.curr_enc_embs            #len4: torch.Size([1, 24, 145, 261]),torch.Size([1, 32, 73, 131]),torch.Size([1, 96, 37, 66]),torch.Size([1, 256, 37, 66])

    def decode_current_logits(self, output_size=None):
        all_logits = []
        for lstn_engine in self.lstn_engines:
            all_logits.append(lstn_engine.decode_current_logits(output_size))
        pred_id_logits = self.soft_logit_aggregation(all_logits)
        return pred_id_logits

    def update_memory(self, curr_mask):
        separated_masks, _ = self.separate_mask(curr_mask, self.obj_nums)           #separated_masks：torch.Size([1, 1, 577, 1041])，curr_mask：torch.Size([1, 1, 577, 1041])
        for lstn_engine, separated_mask in zip(self.lstn_engines, separated_masks):
            lstn_engine.update_short_term_memory(separated_mask)

    def update_size(self):
        self.input_size_2d = self.lstn_engines[0].input_size_2d
        self.enc_size_2d = self.lstn_engines[0].enc_size_2d
        self.enc_hw = self.lstn_engines[0].enc_hw

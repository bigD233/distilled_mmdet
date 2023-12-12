import copy
from typing import Dict, List, Tuple, Optional, Union
import random
import torch
from mmcv.ops import MultiScaleDeformableAttention, batched_nms
from mmdet.models import (DINO, MLP, DeformableDETR,
                          DeformableDetrTransformerDecoder, DINOHead,
                          coordinate_to_encoding, inverse_sigmoid)
from mmdet.models.layers import (CdnQueryGenerator, DeformableDetrTransformerEncoder,
                      DinoTransformerDecoder, SinePositionalEncoding)
from mmdet.models.utils import multi_apply
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy
from mmdet.utils import InstanceList, OptInstanceList, reduce_mean
from mmengine.model import bias_init_with_prob, constant_init
from torch import Tensor, nn
from torch.nn.init import normal_

from .utils import AuxLossModule, align_tensor
from .mm_dino import MultiModalDINO
from .mm_deformable_detr import MultiModalDeformableDETR
from .mm_deformable_detr_layers import MultiModalDeformableDetrTransformerDecoder

class DDQTransformerDecoder(MultiModalDeformableDetrTransformerDecoder):

    def _init_layers(self) -> None:
        super()._init_layers()
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)

    def be_distinct(self, ref_points, query, self_attn_mask, lid):
        num_imgs = len(ref_points)
        dis_start, num_dis = self.cache_dict['dis_query_info']
        # shape of self_attn_mask
        # (batch⋅num_heads, num_queries, embed_dims)
        dis_mask = self_attn_mask[:,dis_start: dis_start + num_dis, \
                   dis_start: dis_start + num_dis]
        # cls_branches from DDQDETRHead
        scores = self.cache_dict['cls_branches'][lid](
            query[:, dis_start:dis_start + num_dis]).sigmoid().max(-1).values
        proposals = ref_points[:, dis_start:dis_start + num_dis]
        proposals = bbox_cxcywh_to_xyxy(proposals)

        attn_mask_list = []
        for img_id in range(num_imgs):
            single_proposals = proposals[img_id]
            single_scores = scores[img_id]
            attn_mask = ~dis_mask[img_id * self.cache_dict['num_heads']][0]
            # distinct query inds in this layer
            ori_index = attn_mask.nonzero().view(-1)
            _, keep_idxs = batched_nms(single_proposals[ori_index],
                                       single_scores[ori_index],
                                       torch.ones(len(ori_index)),
                                       self.cache_dict['dqs_cfg'])

            real_keep_index = ori_index[keep_idxs]

            attn_mask = torch.ones_like(dis_mask[0]).bool()
            # such a attn_mask give best result
            attn_mask[real_keep_index] = False
            attn_mask[:, real_keep_index] = False

            attn_mask = attn_mask[None].repeat(self.cache_dict['num_heads'], 1,
                                               1)
            attn_mask_list.append(attn_mask)
        attn_mask = torch.cat(attn_mask_list)
        self_attn_mask = copy.deepcopy(self_attn_mask)
        self_attn_mask[:, dis_start: dis_start + num_dis, \
            dis_start: dis_start + num_dis] = attn_mask
        # will be used in loss and inference
        self.cache_dict['distinct_query_mask'].append(~attn_mask)
        return self_attn_mask

    def forward(self, query: Tensor, value: Tensor, key_padding_mask: Tensor,
                self_attn_mask: Tensor, reference_points: Tensor,
                spatial_shapes: Tensor, level_start_index: Tensor,
                valid_ratios: Tensor, reg_branches: nn.ModuleList,
                **kwargs) -> Tensor:

        intermediate = []
        intermediate_reference_points = [reference_points]
        self.cache_dict['distinct_query_mask'] = []
        if self_attn_mask is None:
            self_attn_mask = torch.zeros(
                (query.size(1), query.size(1))).bool().cuda()
        # shape is (batch*number_heads, num_queries, num_queries)
        self_attn_mask = self_attn_mask[None].repeat(
            len(query) * self.cache_dict['num_heads'], 1, 1)
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * valid_ratios[:, None]

            query_sine_embed = coordinate_to_encoding(
                reference_points_input[:, :, 0, :],
                num_feats=self.embed_dims // 2)
            query_pos = self.ref_point_head(query_sine_embed)

            query = layer(query,
                          query_pos=query_pos,
                          value=value,
                          key_padding_mask=key_padding_mask,
                          self_attn_mask=self_attn_mask,
                          spatial_shapes=spatial_shapes,
                          level_start_index=level_start_index,
                          valid_ratios=valid_ratios,
                          reference_points=reference_points_input,
                          **kwargs)

            if not self.training:
                tmp = reg_branches[lid](query)
                assert reference_points.shape[-1] == 4
                new_reference_points = tmp + inverse_sigmoid(reference_points,
                                                             eps=1e-3)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()
                if lid < (len(self.layers) - 1):
                    self_attn_mask = self.be_distinct(reference_points, query,
                                                      self_attn_mask, lid)

            else:
                num_dense = self.cache_dict['num_dense_queries']
                tmp_dense = reg_branches[lid](query[:, :-num_dense])
                tmp = self.aux_reg_branches[lid](query[:, -num_dense:])
                tmp = torch.cat([tmp_dense, tmp], dim=1)
                assert reference_points.shape[-1] == 4
                new_reference_points = tmp + inverse_sigmoid(reference_points,
                                                             eps=1e-3)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()
                if lid < (len(self.layers) - 1):
                    self_attn_mask = self.be_distinct(reference_points, query,
                                                      self_attn_mask, lid)

            if self.return_intermediate:
                intermediate.append(self.norm(query))
                intermediate_reference_points.append(new_reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return query, reference_points


@MODELS.register_module()
class ParallelMultiModalDDQDETR(MultiModalDINO):

    def __init__(self,
                 *args,
                 dqs_cfg=dict(type='nms', iou_threshold=0.8),
                 **kwargs):
        self.decoder_cfg = kwargs['decoder']
        self.dqs_cfg = dqs_cfg
        super().__init__(*args, **kwargs)

        # a share dict in all moduls
        # pass some intermediate results and config parameters
        cache_dict = dict()
        for m in self.modules():
            m.cache_dict = cache_dict
        # first element is the start index of matching queries
        # second element is the number of matching queries
        self.cache_dict['dis_query_info'] = [0, 0]

        # mask for distinct queries in each decoder layer
        self.cache_dict['distinct_query_mask'] = []
        # pass to decoder do the dqs
        self.cache_dict['cls_branches'] = self.bbox_head.cls_branches
        # Used to construct the attention mask after dqs
        self.cache_dict['num_heads'] = self.encoder.layers[
            0].self_attn.num_heads
        # pass to decoder to do the dqs
        self.cache_dict['dqs_cfg'] = self.dqs_cfg

    def init_weights(self) -> None:
        super(MultiModalDeformableDETR, self).init_weights()
        for coder in self.encoder, self.decoder:
            for p in coder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        nn.init.xavier_uniform_(self.memory_trans_fc.weight)
        normal_(self.level_embed)

    def _init_layers(self) -> None:
        super(ParallelMultiModalDDQDETR, self)._init_layers()
        self.decoder = DDQTransformerDecoder(**self.decoder_cfg)
        self.query_embedding = None
        self.query_map = nn.Linear(self.embed_dims, self.embed_dims)
        self.query_map_ir = nn.Linear(self.embed_dims, self.embed_dims)

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        def is_substring_of_key(dict_obj, sub_str):
            return any(sub_str in key for key in dict_obj.keys())
        new_keys = [x[0] for x in list(self.named_parameters()) if 'ir' in x[0]]
        reg_keys_6 = [x[0] for x in list(self.named_parameters()) if 'bbox_head.reg_branches.6' in x[0]]
        reg_keys_7 = [x[0] for x in list(self.named_parameters()) if 'bbox_head.reg_branches.7' in x[0]]
        if not is_substring_of_key(state_dict, 'backbone_ir'):
            ori_dict = copy.deepcopy(state_dict)
            num_new_keys = 0
            for new_key in new_keys:
                ori_key = new_key.replace('_ir', '')
                assert ori_key in state_dict.keys()
                state_dict[new_key] = state_dict[ori_key]
                num_new_keys += 1
            for key in reg_keys_6:
                state_dict[new_key.replace('6', '8')] = state_dict[key]
            for key in reg_keys_7:
                state_dict[new_key.replace('7', '9')] = state_dict[key]
            del ori_dict
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        bs, _, c = memory.shape
        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)
        enc_outputs_class = self.bbox_head.cls_branches[ # [2, 7953, 1]
            self.decoder.num_layers](output_memory)
        enc_outputs_coord_unact = self.bbox_head.reg_branches[ # [2, 7953, 4]
            self.decoder.num_layers](output_memory) + output_proposals

        if self.training:
            # -1 is the aux head for the encoder
            dense_enc_outputs_class = self.bbox_head.cls_branches[
                self.decoder.num_layers+1]( # [2, 7953, 1]
                output_memory)
            dense_enc_outputs_coord_unact = self.bbox_head.reg_branches[
                self.decoder.num_layers+1]( # [2, 7953, 4]
                output_memory) + output_proposals

        topk = self.num_queries # 900
        dense_topk = int(topk * 1)

        proposals = enc_outputs_coord_unact.sigmoid() # [2, 7953, 4]
        proposals = bbox_cxcywh_to_xyxy(proposals) # [2, 7953, 4]
        scores = enc_outputs_class.max(-1)[0].sigmoid() # [2, 7953]

        if self.training:
            dense_proposals = dense_enc_outputs_coord_unact.sigmoid() # [2, 7953, 4]
            dense_proposals = bbox_cxcywh_to_xyxy(dense_proposals) # [2, 7953, 4]
            dense_scores = dense_enc_outputs_class.max(-1)[0].sigmoid() # [2, 7953]

        num_imgs = len(scores) # 2
        topk_score = []
        topk_coords_unact = []
        query = []

        dense_topk_score = []
        dense_topk_coords_unact = []
        dense_query = []

        for img_id in range(num_imgs):
            single_proposals = proposals[img_id]
            single_scores = scores[img_id]
            _, keep_idxs = batched_nms(single_proposals, single_scores,
                                       torch.ones(len(single_scores)),
                                       self.cache_dict['dqs_cfg'])
            if self.training:
                dense_single_proposals = dense_proposals[img_id]
                dense_single_scores = dense_scores[img_id]
                # sort according the score
                _, dense_keep_idxs = batched_nms(
                    dense_single_proposals, dense_single_scores,
                    torch.ones(len(dense_single_scores)), None)

                dense_topk_score.append(dense_enc_outputs_class[img_id]
                                        [dense_keep_idxs][:dense_topk])
                dense_topk_coords_unact.append(
                    dense_enc_outputs_coord_unact[img_id][dense_keep_idxs]
                    [:dense_topk])

            topk_score.append(enc_outputs_class[img_id][keep_idxs][:topk])
            topk_coords_unact.append(
                enc_outputs_coord_unact[img_id][keep_idxs][:topk])

            map_memory = self.query_map(memory[img_id].detach())
            query.append(map_memory[keep_idxs][:topk]) # [B, 900, 256]
            if self.training:
                dense_query.append(map_memory[dense_keep_idxs][:dense_topk]) # [B, 1350, 256]

        topk_score = align_tensor(topk_score, topk)
        topk_coords_unact = align_tensor(topk_coords_unact, topk)
        query = align_tensor(query, topk)
        if self.training:
            dense_topk_score = align_tensor(dense_topk_score)
            dense_topk_coords_unact = align_tensor(dense_topk_coords_unact)

            dense_query = align_tensor(dense_query)
            num_dense_queries = dense_query.size(1)
        if self.training:
            query = torch.cat([query, dense_query], dim=1) # [B, 2250, 256]
            topk_coords_unact = torch.cat(
                [topk_coords_unact, dense_topk_coords_unact], dim=1)

        topk_anchor = topk_coords_unact.sigmoid()
        if self.training:
            dense_topk_anchor = topk_anchor[:, -num_dense_queries:]
            topk_anchor = topk_anchor[:, :-num_dense_queries]

        topk_coords_unact = topk_coords_unact.detach()

        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1) # [B, 198(dn_query) + 500(query) + 500(dense), 256]
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
            ori_size = dn_mask.size(-1) # 198 + 500
            new_size = dn_mask.size(-1) + num_dense_queries # 198 + 500 + 500

            new_dn_mask = dn_mask.new_ones((new_size, new_size)).bool()
            dense_mask = torch.zeros(num_dense_queries,
                                     num_dense_queries).bool()
            self.cache_dict['dis_query_info'] = [dn_label_query.size(1), topk * 2]

            new_dn_mask[ori_size:, ori_size:] = dense_mask # [1198, 1198]
            new_dn_mask[:ori_size, :ori_size] = dn_mask # [1098, 1098]
            dn_meta['num_dense_queries'] = num_dense_queries
            dn_mask = new_dn_mask
            self.cache_dict['num_dense_queries'] = num_dense_queries * 2
            self.decoder.aux_reg_branches = self.bbox_head.aux_reg_branches

        else:
            self.cache_dict['dis_query_info'] = [0, topk * 2]
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None

        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(query=query, # [2, 2448, 256] 模态之间独立
                                   memory=memory, # [2, 7953, 256] 模态之间独立
                                   reference_points=reference_points, # [2, 2448, 4] 模态之间独立
                                   dn_mask=dn_mask) # [2448, 2448] 模态之间相同
        head_inputs_dict = dict(enc_outputs_class=topk_score, # [2, 900, 1] 各个模态一阶段的anchor输出
                                enc_outputs_coord=topk_anchor, # [2, 900, 4] 各个模态一阶段的anchor输出
                                dn_meta=dn_meta) if self.training else dict()
        if self.training:
            head_inputs_dict['aux_enc_outputs_class'] = dense_topk_score
            head_inputs_dict['aux_enc_outputs_coord'] = dense_topk_anchor

        return decoder_inputs_dict, head_inputs_dict

    def pre_decoder_ir(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        bs, _, c = memory.shape
        output_memory, output_proposals = self.gen_encoder_output_proposals_ir(
            memory, memory_mask, spatial_shapes)
        enc_outputs_class = self.bbox_head.cls_branches[ # [2, 7953, 1]
            self.decoder.num_layers+2](output_memory)
        enc_outputs_coord_unact = self.bbox_head.reg_branches[ # [2, 7953, 4]
            self.decoder.num_layers+2](output_memory) + output_proposals

        if self.training:
            # -1 is the aux head for the encoder
            dense_enc_outputs_class = self.bbox_head.cls_branches[
                self.decoder.num_layers+3]( # [2, 7953, 1]
                output_memory)
            dense_enc_outputs_coord_unact = self.bbox_head.reg_branches[
                self.decoder.num_layers+3]( # [2, 7953, 4]
                output_memory) + output_proposals

        topk = self.num_queries # 900
        dense_topk = int(topk * 1)

        proposals = enc_outputs_coord_unact.sigmoid() # [2, 7953, 4]
        proposals = bbox_cxcywh_to_xyxy(proposals) # [2, 7953, 4]
        scores = enc_outputs_class.max(-1)[0].sigmoid() # [2, 7953]

        if self.training:
            dense_proposals = dense_enc_outputs_coord_unact.sigmoid() # [2, 7953, 4]
            dense_proposals = bbox_cxcywh_to_xyxy(dense_proposals) # [2, 7953, 4]
            dense_scores = dense_enc_outputs_class.max(-1)[0].sigmoid() # [2, 7953]

        num_imgs = len(scores) # 2
        topk_score = []
        topk_coords_unact = []
        query = []

        dense_topk_score = []
        dense_topk_coords_unact = []
        dense_query = []

        for img_id in range(num_imgs):
            single_proposals = proposals[img_id]
            single_scores = scores[img_id]
            _, keep_idxs = batched_nms(single_proposals, single_scores,
                                       torch.ones(len(single_scores)),
                                       self.cache_dict['dqs_cfg'])
            if self.training:
                dense_single_proposals = dense_proposals[img_id]
                dense_single_scores = dense_scores[img_id]
                # sort according the score
                _, dense_keep_idxs = batched_nms(
                    dense_single_proposals, dense_single_scores,
                    torch.ones(len(dense_single_scores)), None)

                dense_topk_score.append(dense_enc_outputs_class[img_id]
                                        [dense_keep_idxs][:dense_topk])
                dense_topk_coords_unact.append(
                    dense_enc_outputs_coord_unact[img_id][dense_keep_idxs]
                    [:dense_topk])

            topk_score.append(enc_outputs_class[img_id][keep_idxs][:topk])
            topk_coords_unact.append(
                enc_outputs_coord_unact[img_id][keep_idxs][:topk])

            map_memory = self.query_map_ir(memory[img_id].detach())
            query.append(map_memory[keep_idxs][:topk]) # [B, 900, 256]
            if self.training:
                dense_query.append(map_memory[dense_keep_idxs][:dense_topk]) # [B, 1350, 256]

        topk_score = align_tensor(topk_score, topk)
        topk_coords_unact = align_tensor(topk_coords_unact, topk)
        query = align_tensor(query, topk)
        if self.training:
            dense_topk_score = align_tensor(dense_topk_score)
            dense_topk_coords_unact = align_tensor(dense_topk_coords_unact)

            dense_query = align_tensor(dense_query)
            num_dense_queries = dense_query.size(1)
        if self.training:
            query = torch.cat([query, dense_query], dim=1) # [B, 2250, 256]
            topk_coords_unact = torch.cat(
                [topk_coords_unact, dense_topk_coords_unact], dim=1)

        topk_anchor = topk_coords_unact.sigmoid()
        if self.training:
            dense_topk_anchor = topk_anchor[:, -num_dense_queries:]
            topk_anchor = topk_anchor[:, :-num_dense_queries]

        topk_coords_unact = topk_coords_unact.detach()

        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator_ir(batch_data_samples)
            # query = torch.cat([dn_label_query, query], dim=1) # [B, 2250 + 198, 256]
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
            reference_points = topk_coords_unact
            ori_size = dn_mask.size(-1) # 900 + 198
            new_size = dn_mask.size(-1) + num_dense_queries # 900 + 198 + 1350

            new_dn_mask = dn_mask.new_ones((new_size, new_size)).bool()
            dense_mask = torch.zeros(num_dense_queries,
                                     num_dense_queries).bool()
            self.cache_dict['dis_query_info'] = [dn_label_query.size(1), topk * 2]

            new_dn_mask[ori_size:, ori_size:] = dense_mask
            new_dn_mask[:ori_size, :ori_size] = dn_mask
            dn_meta['num_dense_queries'] = num_dense_queries 
            # ignore the dn mask
            dn_mask = new_dn_mask[dn_label_query.size(1): , dn_label_query.size(1):]
            self.cache_dict['num_dense_queries'] = num_dense_queries * 2
            self.decoder.aux_reg_branches = self.bbox_head.aux_reg_branches

        else:
            self.cache_dict['dis_query_info'] = [0, topk * 2]
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None

        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(query_ir=query, # [2, 2448, 256]
                                   memory_ir=memory, # [2, 7953, 256]
                                   reference_points_ir=reference_points, # [2, 2448, 4]
                                   dn_mask_ir=dn_mask) # [2448, 2448]
        head_inputs_dict = dict(enc_outputs_class_ir=topk_score, # [2, 900, 1]
                                enc_outputs_coord_ir=topk_anchor, # [2, 900, 4]
                                dn_meta=dn_meta) if self.training else dict()
        if self.training:
            head_inputs_dict['aux_enc_outputs_class_ir'] = dense_topk_score
            head_inputs_dict['aux_enc_outputs_coord_ir'] = dense_topk_anchor

        return decoder_inputs_dict, head_inputs_dict


    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        """Forward process of Transformer.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.
        The difference is that the ground truth in `batch_data_samples` is
        required for the `pre_decoder` to prepare the query of DINO.
        Additionally, DINO inherits the `pre_transformer` method and the
        `forward_encoder` method of MultiModalDeformableDETR. More details about the
        two methods can be found in `mmdet/detector/deformable_detr.py`.

        Args:
            img_feats (tuple[Tensor]): Tuple of feature maps from neck. Each
                feature map has shape (bs, dim, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            dict: The dictionary of bbox_head function inputs, which always
            includes the `hidden_states` of the decoder output and may contain
            `references` including the initial and intermediate references.
        """
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats[0], batch_data_samples)
        encoder_inputs_dict_ir, _ = self.pre_transformer(
            img_feats[1], batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)
        encoder_outputs_dict_ir = self.forward_encoder_ir(**encoder_inputs_dict_ir)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        tmp_dec_in_ir, head_inputs_dict_ir = self.pre_decoder_ir(
            **encoder_outputs_dict_ir, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)
        decoder_inputs_dict.update(tmp_dec_in_ir)
        if self.training:
            head_inputs_dict['dn_meta']['num_dense_queries'] *= 2 
            head_inputs_dict_ir['dn_meta']['num_dense_queries'] *= 2 
        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        head_inputs_dict.update(head_inputs_dict_ir)
        return head_inputs_dict

    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor, spatial_shapes: Tensor,
                        level_start_index: Tensor,
                        valid_ratios: Tensor) -> Dict:
        """Forward with Transformer encoder.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            feat (Tensor): Sequential features, has shape (bs, num_feat_points,
                dim).
            feat_mask (Tensor): ByteTensor, the padding mask of the features,
                has shape (bs, num_feat_points).
            feat_pos (Tensor): The positional embeddings of the features, has
                shape (bs, num_feat_points, dim).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).

        Returns:
            dict: The dictionary of encoder outputs, which includes the
            `memory` of the encoder output.
        """
        memory = self.encoder(
            query=feat, # [2, 7953, 256]
            query_pos=feat_pos,
            key_padding_mask=feat_mask,  # for self_attn
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios)
        encoder_outputs_dict = dict(
            memory=memory, # [B, sum(H*W), 256]
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes)
        return encoder_outputs_dict

    def forward_encoder_ir(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor, spatial_shapes: Tensor,
                        level_start_index: Tensor,
                        valid_ratios: Tensor) -> Dict:
        """Forward with Transformer encoder.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            feat (Tensor): Sequential features, has shape (bs, num_feat_points,
                dim).
            feat_mask (Tensor): ByteTensor, the padding mask of the features,
                has shape (bs, num_feat_points).
            feat_pos (Tensor): The positional embeddings of the features, has
                shape (bs, num_feat_points, dim).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).

        Returns:
            dict: The dictionary of encoder outputs, which includes the
            `memory` of the encoder output.
        """
        memory = self.encoder_ir(
            query=feat, # [2, 7953, 256]
            query_pos=feat_pos,
            key_padding_mask=feat_mask,  # for self_attn
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios)
        encoder_outputs_dict = dict(
            memory=memory, # [B, sum(H*W), 256]
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes)
        return encoder_outputs_dict

    def forward_decoder(self,
                        query: Tensor, # [B, 2448, 256]
                        memory: Tensor, # [B, 7953, 256]
                        memory_mask: Tensor, # [B, 7953]
                        reference_points: Tensor, # [B, 2448, 4]
                        query_ir: Tensor, # [B, 2448, 256]
                        memory_ir: Tensor, # [B, 7953, 256]
                        reference_points_ir: Tensor, # [B, 2448, 4]
                        spatial_shapes: Tensor,
                        level_start_index: Tensor,
                        valid_ratios: Tensor,
                        dn_mask: Optional[Tensor] = None,
                        dn_mask_ir: Optional[Tensor] = None) -> Dict:
        """Forward with Transformer decoder.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            query (Tensor): The queries of decoder inputs, has shape
                (bs, num_queries_total, dim), where `num_queries_total` is the
                sum of `num_denoising_queries` and `num_matching_queries` when
                `self.training` is `True`, else `num_matching_queries`.
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points).
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries_total, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            dn_mask (Tensor, optional): The attention mask to prevent
                information leakage from different denoising groups and
                matching parts, will be used as `self_attn_mask` of the
                `self.decoder`, has shape (num_queries_total,
                num_queries_total).
                It is `None` when `self.training` is `False`.

        Returns:
            dict: The dictionary of decoder outputs, which includes the
            `hidden_states` of the decoder output and `references` including
            the initial and intermediate reference_points.
        """
        num_dn, num_ds = self.cache_dict['dis_query_info']
        half_num_ds = int(num_ds/2)
        if self.training:
            half_num_dense = int(self.cache_dict['num_dense_queries'] / 2)
        all_query = torch.cat([query[:, :num_dn, :], # dn query
                               query[:, num_dn : num_dn+half_num_ds, :], # rgb normal query
                               query_ir[:, : half_num_ds, :], # ir normal query
                               query[:, num_dn+half_num_ds :, :], # rgb dense query
                               query_ir[:, half_num_ds :, :], # ir dense query
                               ], dim = 1).contiguous() # [2, 1198(198+500+500) + 1000(500+500), 256]
        all_ref_pts = torch.cat([reference_points[:, :num_dn, :], # dn reference_points
                               reference_points[:, num_dn:num_dn+half_num_ds, :], # rgb normal reference_points
                               reference_points_ir[:, :half_num_ds, :], # ir normal qreference_pointsuery
                               reference_points[:, num_dn+half_num_ds:, :], # rgb dense reference_points
                               reference_points_ir[:, half_num_ds:, :], # ir dense reference_points
                               ], dim = 1).contiguous() # [2, 1198(198+500+500) + 1000(500+500), 256]

        if self.training:
            new_dn_mask = dn_mask.new_ones((dn_mask.size(-1)+dn_mask_ir.size(-1), 
                                            dn_mask.size(-1)+dn_mask_ir.size(-1))).bool()

            new_dn_mask[:num_dn+half_num_ds, 
                        :num_dn+half_num_ds] = dn_mask[:num_dn+half_num_ds, :num_dn+half_num_ds] # 198 + 500
            new_dn_mask[num_dn+half_num_ds : num_dn+num_ds, 
                        num_dn+half_num_ds : num_dn+num_ds] = dn_mask_ir[:half_num_ds, :half_num_ds]  # 198 + 500 + 500
            new_dn_mask[num_dn+num_ds : num_dn+num_ds+half_num_dense, 
                        num_dn+num_ds : num_dn+num_ds+half_num_dense] = dn_mask[num_dn+half_num_ds:, 
                                                                             num_dn+half_num_ds:]  # 198 + 500 + 500 + 500(dense)
            new_dn_mask[num_dn+num_ds+half_num_dense : num_dn+num_ds+half_num_dense*2, 
                        num_dn+num_ds+half_num_dense : num_dn+num_ds+half_num_dense*2] = dn_mask_ir[half_num_ds:, half_num_ds:]  # 198 + 500 + 500 + 500(dense) + 500(dense)
        else:
            new_dn_mask = None

        all_memory = torch.cat([memory, memory_ir], dim = 1) # [2, 15906, 256]

        inter_states, references = self.decoder( # [6, 2, 2448, 256], [7, 2, 2448, 4]
            query=all_query, # [1, 2396, 256]
            value=all_memory, # [1, 15906, 256]
            key_padding_mask=torch.cat([memory_mask, memory_mask], dim=1), # [1, 7953 * 2]
            self_attn_mask=new_dn_mask, # [2396, 2396]
            reference_points=all_ref_pts, # [1, 2396, 4]
            spatial_shapes=spatial_shapes, # [4, 2]
            level_start_index=level_start_index,# [4]
            valid_ratios=valid_ratios, # [1, 4, 2]
            reg_branches=self.bbox_head.reg_branches)

        if len(query) == self.num_queries:
            # NOTE: This is to make sure label_embeding can be involved to
            # produce loss even if there is no denoising query (no ground truth
            # target in this GPU), otherwise, this will raise runtime error in
            # distributed training.
            inter_states[0] += \
                self.dn_query_generator.label_embedding.weight[0, 0] * 0.0

        decoder_outputs_dict = dict(
            hidden_states=inter_states, references=list(references))
        return decoder_outputs_dict
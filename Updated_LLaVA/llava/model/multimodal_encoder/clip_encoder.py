import math

import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
import torch.nn.functional as F
from einops import rearrange
from .utils import split_chessboard, merge_chessboard
import torch, open_clip

class CLIPVisionTower(nn.Module):
    def multiscale_forward(self,model, input, image_features=None, scales=None, img_sizes=None, max_split_size=None,
                           resize_output_to_idx=0, num_prefix_token=0,
                           output_shape='bnc'):

        assert input.dim() == 4, "Input image must be in the shape of BxCxHxW."
        assert input.shape[2] == input.shape[3], "Currently only square images are supported."
        assert output_shape in ['bnc',
                                'bchw'], "Output shape should be either BxNxC (e.g., ViT) or BxCxHxW (e.g., ConvNet)."
        assert output_shape == 'bnc' or num_prefix_token == 0, "For ConvNet there shouldn't be any prefix token."

        b, c, input_size, _ = input.shape

        # image size for each scale
        assert scales is not None or img_sizes is not None, "Please assign either scales or img_sizes."
        img_sizes = img_sizes or [int(input_size * scale) for scale in scales]

        # prepare multiscale inputs
        max_split_size = max_split_size or input_size  # The maximum size of each split of image. Set as the input size by default
        num_splits = [math.ceil(size / max_split_size) for size in img_sizes]  # number of splits each scale
        input_multiscale = []
        for size, num_split in zip(img_sizes, num_splits):
            x = F.interpolate(input.to(torch.float32), size=size, mode='bicubic').to(input.dtype)
            x = split_chessboard(x, num_split=num_split)
            # print("x")
            # print(x.shape)

            input_multiscale.append(x)
        # run feedforward on each scale
        # for x in input_multiscale:
        #     print("x")
        #     print(x.shape)
        # image_forward_outs = self.vision_tower(input.to(device=self.device, dtype=self.dtype), output_hidden_states=True)

        # image_features = self.feature_select(image_forward_outs).to(images.dtype)
        outs11_multiscale = [model(x.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
                           for x in input_multiscale]
        # for y in input_multiscale:
        outs_multiscale = [self.feature_select(x)
                           for x in outs11_multiscale]
        # print(outs_multiscale[1])
        # for out in outs_multiscale:
        #     print("out")
        #     print(out)
        # print(out.shape)
        if num_prefix_token > 0:
            outs_prefix_multiscale = [out[:, :num_prefix_token] for out in outs_multiscale]
            outs_multiscale = [out[:, num_prefix_token:] for out in outs_multiscale]
        if output_shape == 'bnc':
            # for out in outs_multiscale:
            #     print("-----------------")
            #     print(out.last_hidden_state.shape[2])
                # print(rearrange(out.last_hidden_state, 'b (h w) c -> b c h w', h=int(out.last_hidden_state.shape[1]** 0.5),
                #       w=int(out.last_hidden_state.shape[1])).shape)
            # outs_multiscale = [out = out.last_hidden_state.reshpe( for out in outs_multiscale]
            outs_multiscale = [rearrange(out, 'b (h w) c -> b c h w', h=int(out.shape[1] ** 0.5),
                                         w=int(out.shape[1] ** 0.5))
                               for out in outs_multiscale]

        # merge outputs of different splits for each scale separately
        outs_multiscale = [merge_chessboard(out, num_split=num_split) for num_split, out in
                           zip(num_splits, outs_multiscale)]

        # interpolate outputs from different scales and concat together
        output_size = outs_multiscale[resize_output_to_idx].shape[-2]
        out = torch.cat([F.interpolate(outs_multiscale[i].to(torch.float16), size=output_size,
                                       mode='area').to(outs_multiscale[i].dtype)
                         for i in range(len(outs_multiscale))], dim=0)
        print(out.size())

        if output_shape == 'bnc':
            out = rearrange(out, 'b c h w -> b (h w) c')
        if num_prefix_token > 0:
            # take the mean of prefix tokens from different splits for each scale
            outs_prefix_multiscale = [torch.stack(out.split(b, dim=0), dim=0).mean(dim=0) for out in
                                      outs_prefix_multiscale]
            out_prefix_multiscale = torch.cat(outs_prefix_multiscale, dim=-1)
            print(out_prefix_multiscale.size())


        return out

    def clip_interpolate_embeddings(self, image_size=600, patch_size=14):
        """This function helps interpolating positional embeddings during checkpoint loading,
        especially when you want to apply a pre-trained model on images with different resolution.

        Args:
            image_size (int): Image size of the new model.
            patch_size (int): Patch size of the new model.
            model_state (OrderedDict[str, torch.Tensor]): State dict of the pre-trained model.
            interpolation_mode (str): The algorithm used for upsampling. Default: bicubic.
            reset_heads (bool): If true, not copying the state of heads. Default: False.

        Returns:
            OrderedDict[str, torch.Tensor]: A state dict which can be loaded into the new model.
        """
        # Shape of pos_embedding is (1, seq_length, hidden_dim)
        state_dict = self.vision_tower.vision_model.embeddings.position_embedding.state_dict()
        pos_embedding = state_dict['weight']
        pos_embedding = pos_embedding.unsqueeze(0)
        n, seq_length, hidden_dim = pos_embedding.shape
        if n != 1:
            raise ValueError(f"Unexpected position embedding shape: {pos_embedding.shape}")

        new_seq_length = (image_size // patch_size) ** 2 + 1

        # Need to interpolate the weights for the position embedding.
        # We do this by reshaping the positions embeddings to a 2d grid, performing
        # an interpolation in the (h, w) space and then reshaping back to a 1d grid.
        if new_seq_length != seq_length:
            # The class token embedding shouldn't be interpolated so we split it up.
            seq_length -= 1
            new_seq_length -= 1
            pos_embedding_token = pos_embedding[:, :1, :]
            pos_embedding_img = pos_embedding[:, 1:, :]

            # (1, seq_length, hidden_dim) -> (1, hidden_dim, seq_length)
            pos_embedding_img = pos_embedding_img.permute(0, 2, 1)
            seq_length_1d = int(math.sqrt(seq_length))
            torch._assert(seq_length_1d * seq_length_1d == seq_length, "seq_length is not a perfect square!")

            # (1, hidden_dim, seq_length) -> (1, hidden_dim, seq_l_1d, seq_l_1d)
            pos_embedding_img = pos_embedding_img.reshape(1, hidden_dim, seq_length_1d, seq_length_1d)
            new_seq_length_1d = image_size // patch_size

            # Perform interpolation.
            # (1, hidden_dim, seq_l_1d, seq_l_1d) -> (1, hidden_dim, new_seq_l_1d, new_seq_l_1d)
            new_pos_embedding_img = nn.functional.interpolate(
                pos_embedding_img,
                size=new_seq_length_1d,
                mode='bicubic',
                align_corners=True,
            )

            # (1, hidden_dim, new_seq_l_1d, new_seq_l_1d) -> (1, hidden_dim, new_seq_length)
            new_pos_embedding_img = new_pos_embedding_img.reshape(1, hidden_dim, new_seq_length)

            # (1, hidden_dim, new_seq_length) -> (1, new_seq_length, hidden_dim)
            new_pos_embedding_img = new_pos_embedding_img.permute(0, 2, 1)
            new_pos_embedding = torch.cat([pos_embedding_token, new_pos_embedding_img], dim=1)[0]
            state_dict['weight'] = new_pos_embedding
            self.vision_tower.vision_model.embeddings.position_embedding = nn.Embedding(new_seq_length + 1, hidden_dim)
            self.vision_tower.vision_model.embeddings.position_embedding.load_state_dict(state_dict)
            self.vision_tower.vision_model.embeddings.image_size = image_size
            self.vision_tower.vision_model.embeddings.patch_size = patch_size
            self.vision_tower.vision_model.embeddings.position_ids = torch.arange(new_seq_length + 1).expand((1, -1))

    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        # print("image")

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

            # self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
            # self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
            # self.vision_tower.requires_grad_(False)
            # self.clip_interpolate_embeddings(image_size=504, patch_size=14)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name,device_map=device_map)

        # self.vision_tower = CLIPVisionModel.from_pretrained('/data/Instruct_tuning/GeoChat/openai/CLIP-ViT-L-14-laion2B-s32B-b82K-remoteclip',device_map=device_map)
        # self.vision_tower, _, self.image_processor = open_clip.create_model_and_transforms(self.vision_tower_name)
        # pt = torch.load("/data/Instruct_tuning/GeoChat/remoteclip/ViT-L-14.pt", map_location='cpu')
        # pt = torch.load("/data/user3/gzqy/Coca_log/Coca_rotation_Foundation/16w_best/checkpoints/epoch_2.pt", map_location='cpu')
        # pt = torch.load("/data/Instruct_tuning/open_clip-main/logs/LR+HRRandom10tif_6epochWC1V4/checkpoints/epoch_6.pt", map_location='cpu')
        print("dww loading weight------------------")
        # self.vision_tower.load_state_dict(pt)
        # self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, state_dict=pt, device_map=device_map)
        self.vision_tower.requires_grad_(False)
        # self.clip_interpolate_embeddings(image_size=504, patch_size=14)
        # print("dww 512------------------")

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        # print("dww LLAVA336 select layer------")
        image_features = image_forward_outs.hidden_states[self.select_layer]
        # image_features = [image_forward_outs['hidden_states'][index][:, 1:] for index in [-2, 6]]
        # print("image_features")
        # print(image_features.shape)
        # image_features = [image_forward_outs['hidden_states'][index][:, 1:] for index in [-2,-5,-8,-11, 6]]
        # image_features = torch.cat(image_features, dim=0)
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            # print("image")
            # print(images.shape)
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            # print(image_forward_outs.size)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            # image_features = self.multiscale_forward(self.vision_tower, images, scales=[1, 2])
            # print("dww LLAVA336 multi scale?------")
            # print("==================image_features=============")
            # print(image_features.shape)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
#
# class CLIPVisionTowerS2(CLIPVisionTower):
#     def clip_interpolate_embeddings(self, image_size=600, patch_size=14):
#         """This function helps interpolating positional embeddings during checkpoint loading,
#         especially when you want to apply a pre-trained model on images with different resolution.
#
#         Args:
#             image_size (int): Image size of the new model.
#             patch_size (int): Patch size of the new model.
#             model_state (OrderedDict[str, torch.Tensor]): State dict of the pre-trained model.
#             interpolation_mode (str): The algorithm used for upsampling. Default: bicubic.
#             reset_heads (bool): If true, not copying the state of heads. Default: False.
#
#         Returns:
#             OrderedDict[str, torch.Tensor]: A state dict which can be loaded into the new model.
#         """
#         # Shape of pos_embedding is (1, seq_length, hidden_dim)
#         state_dict = self.vision_tower.vision_model.embeddings.position_embedding.state_dict()
#         pos_embedding = state_dict['weight']
#         pos_embedding = pos_embedding.unsqueeze(0)
#         n, seq_length, hidden_dim = pos_embedding.shape
#         if n != 1:
#             raise ValueError(f"Unexpected position embedding shape: {pos_embedding.shape}")
#
#         new_seq_length = (image_size // patch_size) ** 2 + 1
#
#         # Need to interpolate the weights for the position embedding.
#         # We do this by reshaping the positions embeddings to a 2d grid, performing
#         # an interpolation in the (h, w) space and then reshaping back to a 1d grid.
#         if new_seq_length != seq_length:
#             # The class token embedding shouldn't be interpolated so we split it up.
#             seq_length -= 1
#             new_seq_length -= 1
#             pos_embedding_token = pos_embedding[:, :1, :]
#             pos_embedding_img = pos_embedding[:, 1:, :]
#
#             # (1, seq_length, hidden_dim) -> (1, hidden_dim, seq_length)
#             pos_embedding_img = pos_embedding_img.permute(0, 2, 1)
#             seq_length_1d = int(math.sqrt(seq_length))
#             torch._assert(seq_length_1d * seq_length_1d == seq_length, "seq_length is not a perfect square!")
#
#             # (1, hidden_dim, seq_length) -> (1, hidden_dim, seq_l_1d, seq_l_1d)
#             pos_embedding_img = pos_embedding_img.reshape(1, hidden_dim, seq_length_1d, seq_length_1d)
#             new_seq_length_1d = image_size // patch_size
#
#             # Perform interpolation.
#             # (1, hidden_dim, seq_l_1d, seq_l_1d) -> (1, hidden_dim, new_seq_l_1d, new_seq_l_1d)
#             new_pos_embedding_img = nn.functional.interpolate(
#                 pos_embedding_img,
#                 size=new_seq_length_1d,
#                 mode='bicubic',
#                 align_corners=True,
#             )
#
#             # (1, hidden_dim, new_seq_l_1d, new_seq_l_1d) -> (1, hidden_dim, new_seq_length)
#             new_pos_embedding_img = new_pos_embedding_img.reshape(1, hidden_dim, new_seq_length)
#
#             # (1, hidden_dim, new_seq_length) -> (1, new_seq_length, hidden_dim)
#             new_pos_embedding_img = new_pos_embedding_img.permute(0, 2, 1)
#             new_pos_embedding = torch.cat([pos_embedding_token, new_pos_embedding_img], dim=1)[0]
#             state_dict['weight'] = new_pos_embedding
#             self.vision_tower.vision_model.embeddings.position_embedding = nn.Embedding(new_seq_length + 1, hidden_dim)
#             self.vision_tower.vision_model.embeddings.position_embedding.load_state_dict(state_dict)
#             self.vision_tower.vision_model.embeddings.image_size = image_size
#             self.vision_tower.vision_model.embeddings.patch_size = patch_size
#             self.vision_tower.vision_model.embeddings.position_ids = torch.arange(new_seq_length + 1).expand((1, -1))
#
#     def __init__(self, vision_tower, args, delay_load=False):
#         super().__init__(vision_tower, args, delay_load)
#
#         self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
#         self.s2_scales = list(map(int, self.s2_scales.split(',')))
#         self.s2_scales.sort()
#         self.s2_split_size = self.s2_scales[0]
#         self.s2_image_size = self.s2_scales[-1]
#
#         try:
#             from s2wrapper import forward as multiscale_forward
#         except ImportError:
#             raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
#         self.multiscale_forward = multiscale_forward
#
#         # change resize/crop size in preprocessing to the largest image size in s2_scale
#         if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
#             self.image_processor.size['shortest_edge'] = self.s2_image_size
#             self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size
#
#             self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
#             self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
#             self.vision_tower.requires_grad_(False)
#             self.clip_interpolate_embeddings(image_size=512, patch_size=14)
#
#     def load_model(self, device_map=None):
#         if self.is_loaded:
#             print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
#             return
#
#         self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
#         self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
#         self.vision_tower.requires_grad_(False)
#         self.clip_interpolate_embeddings(image_size=512, patch_size=14)
#
#         self.image_processor.size['shortest_edge'] = self.s2_image_size
#         self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size
#
#         self.is_loaded = True
#
#     @torch.no_grad()
#     def forward_feature(self, images):
#         image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
#         image_features = self.feature_select(image_forward_outs).to(images.dtype)
#         return image_features
#
#     @torch.no_grad()
#     def forward(self, images):
#         if type(images) is list:
#             image_features = []
#             for image in images:
#                 image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
#                 image_features.append(image_feature)
#         else:
#             image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
#
#         return image_features
#
#     @property
#     def hidden_size(self):
#         return self.config.hidden_size * len(self.s2_scales)

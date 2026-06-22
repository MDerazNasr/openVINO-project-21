# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Mohamed Deraz Nasr] in [2026]. 
# Modification: Native Patching for OpenVINO Export (v2).

from typing import Optional
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from unifolm_vla.model.modules.action_model.flow_matching_modules.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)

from unifolm_vla.model.modules.action_model.flow_matching_modules.cross_attention_dit import DiT
from unifolm_vla.rlds_dataloader.constants import ACTION_DIM, PROPRIO_DIM, NUM_ACTIONS_CHUNK


def _trace_tensor(name, x):
    if x is None:
        print(f"[TRACE] {name}: None")
        return
    print(
        f"[TRACE] {name}: shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}"
    )


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        B, T, _ = actions.shape
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        
        a_emb = self.layer1(actions)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.layer2(x))
        x = self.layer3(x)
        return x


class FlowmatchingActionHead_v2(nn.Module):
    """
    V2: Natively patched to remove stochastic dependence and BatchFeature.
    """
    def __init__(
        self,
        full_config: Optional[dict] = None,
    ):
        super().__init__()
        self.config = full_config.framework.action_model
        self.action_dim = ACTION_DIM
        self.action_horizon = NUM_ACTIONS_CHUNK
        self.num_inference_timesteps = self.config.num_inference_timesteps
        self.num_timestep_buckets = self.config.num_timestep_buckets

        # Actions are projected to input_embedding_dim (1536) for the Transformer
        self.action_encoder = ActionEncoder(
            self.action_dim, self.config.input_embedding_dim
        )
        self.state_encoder = MLP(
            PROPRIO_DIM, self.config.hidden_size, self.config.input_embedding_dim
        )
        self.future_tokens = nn.Embedding(
            self.config.num_target_vision_tokens, self.config.input_embedding_dim
        )
        
        if self.config.add_pos_embed:
            self.position_embedding = nn.Embedding(self.config.max_seq_len, self.config.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # DiT model operates on input_embedding_dim (1536) but outputs hidden_size (1024)
        self.model = DiT(**self.config.diffusion_model_cfg)
        
        # Action decoder takes the DiT output (1024) and projects to action_dim (7)
        self.action_decoder = MLP(
            self.config.hidden_size,
            self.config.hidden_size,
            self.action_dim,
        )

        self.beta_dist = Beta(self.config.noise_beta_alpha, self.config.noise_beta_beta)

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> dict:
        return batch

    def forward(self, vl_embs: torch.Tensor, actions: torch.Tensor, state: torch.Tensor = None, t: torch.Tensor = None):
        device = vl_embs.device
        if t is None:
            noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
            t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
            t = t[:, None, None]  
            noisy_trajectory = (1 - t) * noise + t * actions
            velocity = actions - noise
        else:
            noisy_trajectory = actions
            t = t[:, None, None]

        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized)

        state_features = self.state_encoder(state) if state is not None else None

        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1) \
            if state_features is not None else torch.cat((future_tokens, action_features), dim=1)
        
        model_output = self.model( 
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            timestep=t_discretized,
        )
        pred = self.action_decoder(model_output)
        pred_actions = pred[:, -actions.shape[1] :]

        return pred_actions

    @torch.no_grad()
    def predict_action(self, vl_embs: torch.Tensor, state: torch.Tensor = None, initial_noise: torch.Tensor = None) -> torch.Tensor:
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        
        if initial_noise is not None:
            actions = initial_noise
        else:
            actions = torch.randn(
                size=(batch_size, self.action_horizon, self.action_dim),
                dtype=vl_embs.dtype,
                device=device,
            )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps
        
        state_features = self.state_encoder(state) if state is not None else None

        for t in range(num_steps):
            t_cont = t / float(num_steps)
            t_tensor = torch.full((batch_size,), t_cont, device=device, dtype=vl_embs.dtype)
            pred_velocity = self.forward(vl_embs, actions, state, t=t_tensor)
            actions = actions + dt * pred_velocity
        return actions

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

import torch
import torch.nn as nn

class SingleStepDiTWrapper(nn.Module):
    def __init__(self, action_model):
        super().__init__()
        self.action_model = action_model

    def forward(self, vl_embs, actions, state, timesteps_tensor):
        action_features = self.action_model.action_encoder(actions, timesteps_tensor)

        if self.action_model.config.add_pos_embed:
            pos_ids = torch.arange(
                action_features.shape[1],
                dtype=torch.long,
                device=action_features.device,
            )
            pos_embs = self.action_model.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        state_features = self.action_model.state_encoder(state) if state is not None else None
        future_tokens = self.action_model.future_tokens.weight.unsqueeze(0).expand(
            vl_embs.shape[0], -1, -1
        )

        if state_features is not None:
            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)
        else:
            sa_embs = torch.cat((future_tokens, action_features), dim=1)

        model_output = self.action_model.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            timestep=timesteps_tensor,
        )

        pred = self.action_model.action_decoder(model_output)
        pred_velocity = pred[:, -self.action_model.action_horizon :]
        return pred_velocity

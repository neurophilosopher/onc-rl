"""LSTM policy architectures for ARS and PPO, with optional learned input projection."""
import torch
import torch.nn as nn


class LSTMPolicyARS(nn.Module):
    """LSTM policy with tanh-scaled action output for ARS training."""
    def __init__(self, obs_dim, action_dim, hidden_dim=11,
                 action_low=-1.0, action_high=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_low = action_low
        self.action_high = action_high
        self.input_proj = nn.Linear(obs_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, action_dim)

    def reset_hidden(self, batch_size=1):
        return (torch.zeros(1, batch_size, self.hidden_dim),
                torch.zeros(1, batch_size, self.hidden_dim))

    def forward(self, obs, hidden):
        x = self.input_proj(obs)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        lstm_out, new_hidden = self.lstm(x, hidden)
        lstm_out = lstm_out[:, -1, :] if lstm_out.dim() == 3 else lstm_out
        action = torch.tanh(self.output_proj(lstm_out))
        action = self.action_low + (action + 1.0) * 0.5 * (self.action_high - self.action_low)
        return action, new_hidden

    def flatten_params(self):
        return torch.cat([p.data.view(-1) for p in self.parameters()])

    def unflatten_params(self, flat):
        idx = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(flat[idx:idx+n].view(p.shape))
            idx += n


class LSTMPolicyARSProjected(nn.Module):
    """LSTM-ARS policy with a learnable obs_dim -> project_dim bottleneck before the LSTM."""
    def __init__(self, obs_dim, action_dim, hidden_dim=12, project_dim=2,
                 action_low=-1.0, action_high=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_low = action_low
        self.action_high = action_high
        self.projection = nn.Linear(obs_dim, project_dim)
        self.input_proj = nn.Linear(project_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, action_dim)

    def reset_hidden(self, batch_size=1):
        return (torch.zeros(1, batch_size, self.hidden_dim),
                torch.zeros(1, batch_size, self.hidden_dim))

    def forward(self, obs, hidden):
        x = self.projection(obs)
        x = self.input_proj(x)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        lstm_out, new_hidden = self.lstm(x, hidden)
        lstm_out = lstm_out[:, -1, :] if lstm_out.dim() == 3 else lstm_out
        action = torch.tanh(self.output_proj(lstm_out))
        action = self.action_low + (action + 1.0) * 0.5 * (self.action_high - self.action_low)
        return action, new_hidden

    def flatten_params(self):
        return torch.cat([p.data.view(-1) for p in self.parameters()])

    def unflatten_params(self, flat):
        idx = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(flat[idx:idx+n].view(p.shape))
            idx += n


class LSTMPolicyPPO(nn.Module):
    """Separate actor/critic LSTMs with state-independent log-std."""
    def __init__(self, obs_dim, action_dim, hidden_dim=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.actor_input = nn.Linear(obs_dim, hidden_dim)
        self.actor_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        self.critic_input = nn.Linear(obs_dim, hidden_dim)
        self.critic_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.critic_value = nn.Linear(hidden_dim, 1)

    def reset_hidden(self, batch_size=1):
        actor_h = (torch.zeros(1, batch_size, self.hidden_dim),
                   torch.zeros(1, batch_size, self.hidden_dim))
        critic_h = (torch.zeros(1, batch_size, self.hidden_dim),
                    torch.zeros(1, batch_size, self.hidden_dim))
        return (actor_h, critic_h)

    def forward(self, obs, hidden):
        actor_h, critic_h = hidden
        ax = torch.relu(self.actor_input(obs)).unsqueeze(1)
        a_out, new_actor_h = self.actor_lstm(ax, actor_h)
        action_mean = self.actor_mean(a_out.squeeze(1))
        cx = torch.relu(self.critic_input(obs)).unsqueeze(1)
        c_out, new_critic_h = self.critic_lstm(cx, critic_h)
        value = self.critic_value(c_out.squeeze(1)).squeeze(-1)
        return action_mean, value, (new_actor_h, new_critic_h)

    def evaluate_sequences(self, obs_seq, actions_seq, dones_seq, initial_hidden):
        """Re-run the full rollout, resetting hidden state at episode boundaries."""
        T = obs_seq.shape[0]
        actor_h, critic_h = initial_hidden
        all_action_means, all_values = [], []
        for t in range(T):
            if t > 0 and dones_seq[t - 1] > 0.5:
                actor_h = (torch.zeros_like(actor_h[0]), torch.zeros_like(actor_h[1]))
                critic_h = (torch.zeros_like(critic_h[0]), torch.zeros_like(critic_h[1]))
            obs_t = obs_seq[t:t+1].unsqueeze(1)
            ax = torch.relu(self.actor_input(obs_t))
            a_out, actor_h = self.actor_lstm(ax, actor_h)
            action_mean = self.actor_mean(a_out.squeeze(1))
            all_action_means.append(action_mean)
            cx = torch.relu(self.critic_input(obs_t))
            c_out, critic_h = self.critic_lstm(cx, critic_h)
            value = self.critic_value(c_out.squeeze(1)).squeeze(-1)
            all_values.append(value)
        action_means = torch.cat(all_action_means, dim=0)
        values = torch.cat(all_values, dim=0)
        std = torch.exp(self.actor_log_std)
        dist = torch.distributions.Normal(action_means, std)
        new_log_probs = dist.log_prob(actions_seq).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()
        return new_log_probs, values, entropy


class LSTMPolicyPPOProjected(nn.Module):
    """PPO policy with separate learnable obs_dim -> project_dim bottlenecks for actor and critic."""
    def __init__(self, obs_dim, action_dim, hidden_dim=12, project_dim=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.actor_projection = nn.Linear(obs_dim, project_dim)
        self.actor_input = nn.Linear(project_dim, hidden_dim)
        self.actor_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        self.critic_projection = nn.Linear(obs_dim, project_dim)
        self.critic_input = nn.Linear(project_dim, hidden_dim)
        self.critic_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.critic_value = nn.Linear(hidden_dim, 1)

    def reset_hidden(self, batch_size=1):
        actor_h = (torch.zeros(1, batch_size, self.hidden_dim),
                   torch.zeros(1, batch_size, self.hidden_dim))
        critic_h = (torch.zeros(1, batch_size, self.hidden_dim),
                    torch.zeros(1, batch_size, self.hidden_dim))
        return (actor_h, critic_h)

    def forward(self, obs, hidden):
        actor_h, critic_h = hidden
        ax = torch.relu(self.actor_input(self.actor_projection(obs))).unsqueeze(1)
        a_out, new_actor_h = self.actor_lstm(ax, actor_h)
        action_mean = self.actor_mean(a_out.squeeze(1))
        cx = torch.relu(self.critic_input(self.critic_projection(obs))).unsqueeze(1)
        c_out, new_critic_h = self.critic_lstm(cx, critic_h)
        value = self.critic_value(c_out.squeeze(1)).squeeze(-1)
        return action_mean, value, (new_actor_h, new_critic_h)

    def evaluate_sequences(self, obs_seq, actions_seq, dones_seq, initial_hidden):
        """Re-run the full rollout, resetting hidden state at episode boundaries."""
        T = obs_seq.shape[0]
        actor_h, critic_h = initial_hidden
        all_action_means, all_values = [], []
        for t in range(T):
            if t > 0 and dones_seq[t - 1] > 0.5:
                actor_h = (torch.zeros_like(actor_h[0]), torch.zeros_like(actor_h[1]))
                critic_h = (torch.zeros_like(critic_h[0]), torch.zeros_like(critic_h[1]))
            obs_t = obs_seq[t:t+1].unsqueeze(1)
            ax = torch.relu(self.actor_input(self.actor_projection(obs_t)))
            a_out, actor_h = self.actor_lstm(ax, actor_h)
            action_mean = self.actor_mean(a_out.squeeze(1))
            all_action_means.append(action_mean)
            cx = torch.relu(self.critic_input(self.critic_projection(obs_t)))
            c_out, critic_h = self.critic_lstm(cx, critic_h)
            value = self.critic_value(c_out.squeeze(1)).squeeze(-1)
            all_values.append(value)
        action_means = torch.cat(all_action_means, dim=0)
        values = torch.cat(all_values, dim=0)
        std = torch.exp(self.actor_log_std)
        dist = torch.distributions.Normal(action_means, std)
        new_log_probs = dist.log_prob(actions_seq).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()
        return new_log_probs, values, entropy
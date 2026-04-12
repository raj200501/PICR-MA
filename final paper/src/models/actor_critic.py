from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.envs.observation_slicing import ObservationSlicing, split_self_teammates


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=1, keepdims=True)


def _init_matrix(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    scale = np.sqrt(2.0 / max(1, fan_in))
    return rng.normal(0.0, scale, size=(fan_in, fan_out)).astype(np.float32)


@dataclass
class ForwardPass:
    probs: np.ndarray
    values: np.ndarray
    logits: np.ndarray
    cache: dict


class BaseActorCritic:
    kind = "base"

    def action_distribution(self, observations: np.ndarray) -> np.ndarray:
        return self.forward(observations).probs

    def act(
        self,
        observations: np.ndarray,
        rng: np.random.Generator,
        greedy: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        forward = self.forward(observations)
        probs = forward.probs
        if greedy:
            actions = np.argmax(probs, axis=1)
        else:
            actions = np.array(
                [rng.choice(self.action_dim, p=prob) for prob in probs],
                dtype=np.int64,
            )
        log_probs = np.log(probs[np.arange(len(actions)), actions] + 1e-8)
        return actions, log_probs, forward.values

    def policy_value_backward(
        self,
        forward: ForwardPass,
        dlogits: np.ndarray,
        dvalues: np.ndarray,
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def zero_grads(self) -> dict[str, np.ndarray]:
        return {name: np.zeros_like(value) for name, value in self.parameters().items()}

    def add_grads(
        self,
        accumulator: dict[str, np.ndarray],
        update: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        for name in accumulator:
            accumulator[name] += update[name]
        return accumulator

    def apply_gradients(self, gradients: dict[str, np.ndarray], lr: float) -> None:
        for name, value in self.parameters().items():
            value -= lr * gradients[name]

    def state_dict(self) -> dict[str, np.ndarray]:
        state = {name: value.copy() for name, value in self.parameters().items()}
        state["model_kind"] = np.array(self.kind)
        state["input_dim"] = np.array(self.input_dim)
        state["action_dim"] = np.array(self.action_dim)
        return state

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        for name, value in self.parameters().items():
            value[...] = state[name]

    def parameters(self) -> dict[str, np.ndarray]:
        raise NotImplementedError


class MLPActorCritic(BaseActorCritic):
    kind = "mlp"

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        seed: int = 0,
    ) -> None:
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        rng = np.random.default_rng(seed)
        self.w1 = _init_matrix(rng, input_dim, hidden_dim)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.w_policy = _init_matrix(rng, hidden_dim, action_dim)
        self.b_policy = np.zeros(action_dim, dtype=np.float32)
        self.w_value = _init_matrix(rng, hidden_dim, 1)
        self.b_value = np.zeros(1, dtype=np.float32)

    def parameters(self) -> dict[str, np.ndarray]:
        return {
            "w1": self.w1,
            "b1": self.b1,
            "w_policy": self.w_policy,
            "b_policy": self.b_policy,
            "w_value": self.w_value,
            "b_value": self.b_value,
        }

    def forward(self, observations: np.ndarray) -> ForwardPass:
        x = np.asarray(observations, dtype=np.float32)
        hidden_pre = x @ self.w1 + self.b1
        hidden = np.tanh(hidden_pre)
        logits = hidden @ self.w_policy + self.b_policy
        values = (hidden @ self.w_value + self.b_value).reshape(-1)
        probs = _softmax(logits)
        return ForwardPass(
            probs=probs,
            values=values,
            logits=logits,
            cache={"x": x, "hidden_pre": hidden_pre, "hidden": hidden},
        )

    def policy_value_backward(
        self,
        forward: ForwardPass,
        dlogits: np.ndarray,
        dvalues: np.ndarray,
    ) -> dict[str, np.ndarray]:
        hidden = forward.cache["hidden"]
        x = forward.cache["x"]

        grads = self.zero_grads()
        grads["w_policy"] = hidden.T @ dlogits
        grads["b_policy"] = np.sum(dlogits, axis=0)
        grads["w_value"] = hidden.T @ dvalues[:, None]
        grads["b_value"] = np.array([np.sum(dvalues)], dtype=np.float32)

        dhidden = dlogits @ self.w_policy.T + dvalues[:, None] @ self.w_value.T
        dpre = dhidden * (1.0 - hidden**2)
        grads["w1"] = x.T @ dpre
        grads["b1"] = np.sum(dpre, axis=0)
        return grads


class DeepSetsActorCritic(BaseActorCritic):
    kind = "deepsets"

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        slicing: ObservationSlicing,
        hidden_dim: int = 64,
        set_dim: int = 32,
        seed: int = 0,
    ) -> None:
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.set_dim = set_dim
        self.slicing = slicing
        self.self_dim = len(slicing.self_indices)
        rng = np.random.default_rng(seed)
        self.w_set = _init_matrix(rng, slicing.block_dim, set_dim)
        self.b_set = np.zeros(set_dim, dtype=np.float32)
        self.w1 = _init_matrix(rng, self.self_dim + set_dim, hidden_dim)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.w_policy = _init_matrix(rng, hidden_dim, action_dim)
        self.b_policy = np.zeros(action_dim, dtype=np.float32)
        self.w_value = _init_matrix(rng, hidden_dim, 1)
        self.b_value = np.zeros(1, dtype=np.float32)

    def parameters(self) -> dict[str, np.ndarray]:
        return {
            "w_set": self.w_set,
            "b_set": self.b_set,
            "w1": self.w1,
            "b1": self.b1,
            "w_policy": self.w_policy,
            "b_policy": self.b_policy,
            "w_value": self.w_value,
            "b_value": self.b_value,
        }

    def forward(self, observations: np.ndarray) -> ForwardPass:
        obs = np.asarray(observations, dtype=np.float32)
        self_features = np.stack(
            [split_self_teammates(row, self.slicing)[0] for row in obs],
            axis=0,
        )
        teammate_blocks = np.stack(
            [split_self_teammates(row, self.slicing)[1] for row in obs],
            axis=0,
        )
        set_pre = teammate_blocks @ self.w_set + self.b_set
        set_hidden = np.tanh(set_pre)
        set_agg = np.mean(set_hidden, axis=1)
        encoder_input = np.concatenate([self_features, set_agg], axis=1)
        hidden_pre = encoder_input @ self.w1 + self.b1
        hidden = np.tanh(hidden_pre)
        logits = hidden @ self.w_policy + self.b_policy
        values = (hidden @ self.w_value + self.b_value).reshape(-1)
        probs = _softmax(logits)
        return ForwardPass(
            probs=probs,
            values=values,
            logits=logits,
            cache={
                "self_features": self_features,
                "teammate_blocks": teammate_blocks,
                "set_pre": set_pre,
                "set_hidden": set_hidden,
                "set_agg": set_agg,
                "encoder_input": encoder_input,
                "hidden": hidden,
            },
        )

    def policy_value_backward(
        self,
        forward: ForwardPass,
        dlogits: np.ndarray,
        dvalues: np.ndarray,
    ) -> dict[str, np.ndarray]:
        hidden = forward.cache["hidden"]
        encoder_input = forward.cache["encoder_input"]
        set_hidden = forward.cache["set_hidden"]
        teammate_blocks = forward.cache["teammate_blocks"]

        grads = self.zero_grads()
        grads["w_policy"] = hidden.T @ dlogits
        grads["b_policy"] = np.sum(dlogits, axis=0)
        grads["w_value"] = hidden.T @ dvalues[:, None]
        grads["b_value"] = np.array([np.sum(dvalues)], dtype=np.float32)

        dhidden = dlogits @ self.w_policy.T + dvalues[:, None] @ self.w_value.T
        dpre = dhidden * (1.0 - hidden**2)
        grads["w1"] = encoder_input.T @ dpre
        grads["b1"] = np.sum(dpre, axis=0)

        dencoder_input = dpre @ self.w1.T
        dset_agg = dencoder_input[:, self.self_dim :]
        dset_hidden = np.repeat(
            (dset_agg / max(1, self.slicing.n_teammates))[:, None, :],
            self.slicing.n_teammates,
            axis=1,
        )
        dset_pre = dset_hidden * (1.0 - set_hidden**2)
        grads["w_set"] = np.einsum("btd,bth->dh", teammate_blocks, dset_pre)
        grads["b_set"] = np.sum(dset_pre, axis=(0, 1))
        return grads


def build_actor_critic(
    kind: str,
    input_dim: int,
    action_dim: int,
    seed: int,
    slicing: ObservationSlicing | None = None,
    hidden_dim: int = 64,
) -> BaseActorCritic:
    if kind == "deepsets":
        if slicing is None:
            raise ValueError("DeepSets policy requires an ObservationSlicing.")
        return DeepSetsActorCritic(
            input_dim=input_dim,
            action_dim=action_dim,
            slicing=slicing,
            hidden_dim=hidden_dim,
            seed=seed,
        )
    return MLPActorCritic(
        input_dim=input_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        seed=seed,
    )


def load_actor_critic(
    state: dict[str, np.ndarray],
    slicing: ObservationSlicing | None = None,
) -> BaseActorCritic:
    kind = str(state["model_kind"].item())
    input_dim = int(state["input_dim"].item())
    action_dim = int(state["action_dim"].item())
    if kind == "deepsets":
        if slicing is None:
            raise ValueError("Loading a DeepSets model requires the matching slicing.")
        model = DeepSetsActorCritic(
            input_dim=input_dim,
            action_dim=action_dim,
            slicing=slicing,
            hidden_dim=int(state["w1"].shape[1]),
            set_dim=int(state["w_set"].shape[1]),
            seed=0,
        )
    else:
        model = MLPActorCritic(
            input_dim=input_dim,
            action_dim=action_dim,
            hidden_dim=int(state["w1"].shape[1]),
            seed=0,
        )
    model.load_state_dict(state)
    return model

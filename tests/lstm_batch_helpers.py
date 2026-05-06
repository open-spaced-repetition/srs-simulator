from __future__ import annotations

import torch

from simulator.models.lstm_batch import PackedLSTMWeights


def dummy_lstm_weights(
    n_users: int,
    *,
    n_hidden: int = 1,
    n_curves: int = 1,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> PackedLSTMWeights:
    if device is None:
        device = torch.device("cpu")

    def zeros(*shape: int) -> torch.Tensor:
        return torch.zeros(shape, device=device, dtype=dtype)

    def ones(*shape: int) -> torch.Tensor:
        return torch.ones(shape, device=device, dtype=dtype)

    n_input = 5
    return PackedLSTMWeights(
        use_duration_feature=False,
        n_users=n_users,
        n_hidden=n_hidden,
        n_curves=n_curves,
        n_main_inputs=1,
        n_input=n_input,
        n_rnns=2,
        input_mean=zeros(n_users, 1),
        input_std=ones(n_users, 1),
        process_0_weight=zeros(n_users, n_hidden, n_input),
        process_0_bias=zeros(n_users, n_hidden),
        process_2_weight=ones(n_users, n_hidden),
        process_3_weight=zeros(n_users, n_hidden, n_hidden),
        process_3_bias=zeros(n_users, n_hidden),
        process_5_ln_weight=ones(n_users, n_hidden),
        process_5_lstm_w_ih=zeros(n_users, 4 * n_hidden, n_hidden),
        process_5_lstm_w_hh=zeros(n_users, 4 * n_hidden, n_hidden),
        process_5_lstm_b_ih=zeros(n_users, 4 * n_hidden),
        process_5_lstm_b_hh=zeros(n_users, 4 * n_hidden),
        process_6_ln_weight=ones(n_users, n_hidden),
        process_6_ln_bias=zeros(n_users, n_hidden),
        process_6_lstm_w_ih=zeros(n_users, 4 * n_hidden, n_hidden),
        process_6_lstm_w_hh=zeros(n_users, 4 * n_hidden, n_hidden),
        process_6_lstm_b_ih=zeros(n_users, 4 * n_hidden),
        process_6_lstm_b_hh=zeros(n_users, 4 * n_hidden),
        process_7_ln1_weight=ones(n_users, n_hidden),
        process_7_fc1_weight=zeros(n_users, n_hidden, n_hidden),
        process_7_fc1_bias=zeros(n_users, n_hidden),
        process_7_ln2_weight=ones(n_users, n_hidden),
        process_7_fc2_weight=zeros(n_users, n_hidden, n_hidden),
        process_7_fc2_bias=zeros(n_users, n_hidden),
        process_8_weight=ones(n_users, n_hidden),
        process_9_weight=zeros(n_users, n_hidden, n_hidden),
        process_9_bias=zeros(n_users, n_hidden),
        w_fc_weight=zeros(n_users, n_curves, n_hidden),
        w_fc_bias=zeros(n_users, n_curves),
        s_fc_weight=zeros(n_users, n_curves, n_hidden),
        s_fc_bias=zeros(n_users, n_curves),
        d_fc_weight=zeros(n_users, n_curves, n_hidden),
        d_fc_bias=zeros(n_users, n_curves),
    )

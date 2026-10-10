"""Simulation DGP parameters and model/training hyperparameters.

The DGP (Supplementary Material) has four parts:
  1. Baseline outcome-effect families F_i: quintiles of baseline covariate
     burden. They enter only the latent outcome, scaling the own-treatment
     benefit by m_{F_i}.
  2. Covariates: V_it^(k) = (1-kappa_k)V_{i,t-1}^(k) + mu_k
     + gamma_Y,k(Y_{i,t-1}-c_Y,k) + gamma_YS,k(Y^S_{i,t-1}-c_Y,k) + noise.
  3. Treatment: logistic model in a standardized baseline risk score R_i^0,
     the previous treatment, recent outcome worsening/improvement and the
     lagged covariate burden U_{i,t-1}.
  4. Outcome: latent disease activity C_it, driven by own and neighborhood
     covariate burden and own and spillover treatment histories, smoothed
     into the observed outcome Y_it.
"""
import torch


CONFIG = {
    # ========================================================================
    # DESIGN / NETWORK
    # ========================================================================
    'v_dim': 10,                        # covariate dimension K
    'T': 20,                            # time horizon T
    'treatment_update_interval': 5,     # decision times tau_{1:4} = (1,6,11,16)
    'network_avg_degree': 6,            # mean degree of the random network

    # ========================================================================
    # BASELINE COVARIATES AND OUTCOME
    #   V_i0^(k) ~ N(baseline_v_mean_k, baseline_v_std_k^2), clipped to [0, v_max_k]
    # ========================================================================
    'baseline_v_mean': [52.3, 37.7, 47.4, 34.2, 58.5, 51.0, 31.2, 41.2, 32.5, 46.3],
    'baseline_v_std': [8.0, 6.5, 7.0, 5.5, 7.5, 6.0, 5.0, 5.5, 4.5, 6.5],
    'baseline_y_mean': 50.0,            # Y_0 ~ N(50, 5^2)
    'baseline_y_std': 5.0,
    'v_max': [95.0, 80.0, 85.0, 70.0, 90.0, 78.0, 65.0, 72.0, 60.0, 88.0],  # upper clip for V^(k)
    'y_max': 130.0,                     # Y is clipped to [y_min, y_max]
    'y_min': 0.0,

    # ========================================================================
    # COVARIATE EVOLUTION
    #   V_it^(k) = (1-kappa_k)V_{i,t-1}^(k) + mu_k
    #              + gamma_Y,k(Y_{i,t-1}-c_Y,k) + gamma_YS,k(Y^S_{i,t-1}-c_Y,k) + noise
    #   Y^S is the neighborhood-average outcome; gamma_YS,k = gamma_Y,k / 2,
    #   so own-outcome feedback is stronger than neighborhood feedback.
    #   cov_burden_weights w_k define the burden U_it = sum_k w_k V_it^(k)
    #   (normalized to sum to 1).
    # ========================================================================
    'cov_kappa': [0.07, 0.09, 0.09, 0.15, 0.05, 0.07, 0.21, 0.15, 0.06, 0.11],
    'cov_mu': [0.78, 0.58, 0.75, 0.40, 1.28, 1.10, 0.27, 0.48, 1.16, 0.66],
    'cov_gamma_y': [0.06, 0.10, 0.12, 0.04, 0.11, 0.08, 0.05, 0.05, 0.03, 0.09],
    'cov_gamma_ys': [0.03, 0.05, 0.06, 0.02, 0.055, 0.04, 0.025, 0.025, 0.015, 0.045],
    'cov_c_y': [52.0, 48.0, 49.0, 45.0, 51.0, 48.0, 43.0, 46.0, 40.0, 50.0],
    'cov_sigma_v': [0.10, 0.08, 0.09, 0.07, 0.11, 0.10, 0.13, 0.09, 0.15, 0.10],
    'cov_burden_weights': [1.06, 0.92, 1.16, 0.66, 1.36, 1.14, 0.60, 0.94, 1.02, 0.90],

    # ========================================================================
    # BASELINE OUTCOME-EFFECT FAMILIES: multiplier m_F of the own-treatment
    # benefit for each quintile B1..B5 of baseline burden (lowest to highest)
    # ========================================================================
    'family_multipliers': {
        'B1': 0.50,
        'B2': 1.16,
        'B3': 0.92,
        'B4': 1.20,
        'B5': 1.10,
    },

    # ========================================================================
    # TREATMENT ASSIGNMENT MODEL (at decision times)
    #   logit(pi) = theta_0 + theta_R R_i^0 + theta_p X_prev + theta_w W
    #               + theta_i I + theta_v U_prev + eps^X
    #   W, I: positive and negative parts of the recent outcome change
    #   (worsening, improvement); R_i^0 standardizes
    #   omega_Y0 Y_i0 + omega_V0 bar V_i0 + omega_YS0 Y^S_i0.
    # ========================================================================
    'treatment_theta_0': -3.85,
    'treatment_theta_R': 1.00,
    'treatment_theta_prev': 0.28,
    'treatment_theta_w': 0.19,
    'treatment_theta_i': 0.08,
    'treatment_theta_v': 0.06,
    'risk_omega_Y0': 0.10,               # omega_Y0 in R_i^0
    'risk_omega_V0': 0.18,               # omega_V0 in R_i^0
    'risk_omega_YS0': 0.08,              # omega_YS0 in R_i^0
    'treatment_sigma_X': 0.45,           # sigma_X: logit noise std

    # Treatment-model variant. 'worsening_quadratic' adds theta_w2 W^2 to the
    # logit, so a large recent worsening raises the treatment probability
    # disproportionately; 'baseline' omits it.
    'treatment_nonlinear_variant': 'worsening_quadratic',
    'treatment_theta_w2': 0.05,

    # ========================================================================
    # DYNAMICS: own-treatment history decay
    #   H^X_it = rho_X H^X_{i,t-1} + X_it
    # ========================================================================
    'history_rho_X': 0.63,

    # ========================================================================
    # LATENT DISEASE ACTIVITY C_it
    #   q_1,it = sigmoid(U_{i,t-1} - m_{t-1}),  q_S,it = sigmoid(U^S_{i,t-1} - m_{t-1})
    #   C_it = mu_C + rho_C(C_{i,t-1}-mu_C) + lambda_1 q_1,it + lambda_S q_S,it
    #          - m_{F_i} beta_X (1 - exp(-a_X H^X_it)) - beta_XS (1 - exp(-a_X H^XS_it))
    #          + eps^C
    #   m_t = E[U_t] is the population burden trajectory. lambda_S and beta_XS
    #   are the neighborhood counterparts of lambda_1 and beta_X, at roughly
    #   half their magnitude. beta_XS is not scaled by m_{F_i}: the spillover
    #   treatment effect is homogeneous across families.
    # ========================================================================
    'latent_mu_C': 72.0,
    'latent_rho_C': 0.65,
    'latent_lambda_1': 5.2,
    'latent_lambda_S': 2.5,
    'latent_beta_X': 8.25,
    'latent_beta_XS': 3.0,
    'latent_a_X': 0.85,
    'latent_sigma_C': 0.80,

    # Latent-outcome variant ('baseline' has none of the terms below):
    #   burden_modified:       spillover benefit multiplied by
    #                          1 + eta_B {2 sigmoid(U_{t-1}-m_{t-1}) - 1}
    #   own_spillover_synergy: adds -beta_XD g(H^X_it) g(H^XS_it),
    #                          g(u) = 1 - exp(-a_X u)
    #   burden_synergy:        both terms
    # Main design: own_spillover_synergy with beta_XD = 10.
    # Stress-test designs: burden_modified, own_spillover_synergy and
    # burden_synergy with beta_XD = 4 and treatment variant 'baseline'.
    'latent_nonlinear_variant': 'own_spillover_synergy',
    'latent_spillover_burden_eta': 0.75,    # eta_B
    'latent_beta_XD': 10.0,                 # beta_XD

    # ========================================================================
    # OBSERVED OUTCOME Y_it
    #   Y_it = (1-kappa_Y)Y_{i,t-1} + kappa_Y C_it + eps^Y
    # ========================================================================
    'outcome_kappa_Y': 0.28,
    'outcome_sigma_Y': 0.45,

    # ========================================================================
    # SAMPLE SIZES
    # ========================================================================
    'n_samples_val_ratio': 0.1,        # validation network size as a fraction of n
    'n_samples_monte_carlo': 50000,    # units per Monte Carlo batch for ground truth

    # ========================================================================
    # NEURAL NETWORK ARCHITECTURE (Section 3.1-3.2)
    # ========================================================================
    # LSTM backbone
    'lstm_hidden_dim': 128,         # Hidden state dimension h_{i,t}
    'lstm_num_layers': 2,           # Number of stacked LSTM layers

    # Decision-time representation g_Z(h_t, V_t, V^S_t) -> (Z^X, Z^D, Z^C):
    #   Z^X own-treatment specific, Z^D spillover specific, Z^C shared.
    'z_x_dim': 16,                  # own-treatment latent Z^X
    'z_d_dim': 16,                  # spillover latent Z^D
    'z_c_dim': 32,                  # shared common latent Z^C  (total = 64)

    # Training (Section 3.3)
    'batch_size': 64,
    'learning_rate': 1e-3,
    'l2_lambda': 1e-4,              # L2 penalty weight
    'epochs': 300,
    'dropout': 0.1,
    'early_stopping_patience': 20,  # epochs without validation improvement
    'lr_scheduler_patience': 10,    # epochs before the learning rate is reduced

    # Loss weights
    'weight_treatment_binary': 1.0,   # w_X: own-treatment BCE  L_X
    'weight_spillover': 1.0,          # w_D: spillover BCE       L_D

    # ========================================================================
    # PARALLELISM AND DATA LOADING
    # ========================================================================
    'multi_gpu': False,
    'num_workers': 4,
    'pin_memory': True,
    'persistent_workers': True,
    'prefetch_factor': 2,
    'use_amp': True,                # mixed precision on CUDA
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

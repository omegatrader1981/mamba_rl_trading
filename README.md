# Project Phoenix - DRL Trading System

Mamba-SSM + SAC reinforcement learning for futures trading with microstructure features.

## Current Status: Phase 0 (Pipeline Validation)

---

## Roadmap

### **Phase 0: Infrastructure (Complete)**
- Data pipeline (OHLCV loading, cleaning, regime selection)
- Feature engineering (VWAP, HMM regimes, z-score normalization)
- Mamba + SAC integration
- SageMaker containerization
- **Next:** Smoke test (5K timesteps)

### **Phase 1: Baseline with Lagging Features (2-3 days)**
**Goal:** Validate RL training works, establish baseline performance

**Features (6):**
- `volume_zscore`, `high_low_pct`, `session_vwap_dist`
- `hmm_regime`, `time_sin`, `time_cos`

**Expected Sortino:** 0.5-1.0 (weak features, but proves pipeline)

**Deliverables:**
- First equity curve
- Baseline Sortino/drawdown metrics
- Confirm agent learns (not stuck flat)

---

### **Phase 2: OFI Microstructure Features (2-3 weeks)**
**Goal:** Add real edge with order flow imbalance

**Data:** MBP-10 from Databento (1-3 months, ~$400)

**New Features (11-13 total):**
- `ofi_1min_zscore` (1s L2 to 1min aggregation)
- `ofi_3min_cumulative_zscore`
- `ofi_persistence_zscore` (KEY: trend/hold signal)
- `microprice_dev_zscore` (leading indicator)
- `microprice_5min_avg_zscore`
- `liquidity_consumption_zscore`
- `spread_zscore`
- `atr_zscore`
- `hmm_regime` (one-hot: 3 dims)

**Expected Sortino:** 1.5-2.5

**Preprocessing:** 1s L2 snapshots to compute OFI to aggregate to 1min bars

**Deliverables:**
- OFI feature pipeline
- 1-month validation results
- Ablation study (OFI vs no OFI)

---

### **Phase 3: Optimization & Tuning (1-2 weeks)**
**Goal:** Maximize Sortino via reward shaping and HPO

**Focus Areas:**
1. **Reward function:** Add shaped rewards (win bonus, loss penalty, hold discipline)
2. **Action space:** Test continuous position sizing
3. **Hyperparameters:** Optuna HPO (learning rate, gamma, tau, Mamba layers)
4. **Multi-timeframe:** Add 15min context

**Expected Sortino:** 2.5-3.5

**Deliverables:**
- Tuned reward function
- HPO study results
- 3-month walk-forward validation

---

### **Phase 4: Production Prep (1-2 weeks)**
**Goal:** Prepare for live deployment

**Tasks:**
- Real-time data ingestion (Databento WebSocket)
- Order execution integration (broker API)
- Risk management (position limits, kill switches)
- Monitoring dashboard (P&L, drawdown, latency)
- Paper trading (1-2 weeks validation)

**Deliverables:**
- Live trading infrastructure
- Paper trading results
- Risk management system

---

### **Phase 5: Live Deployment**
**Goal:** Go live with 1 contract, scale if profitable

**Criteria for Go-Live:**
- Paper trading Sortino > 2.0
- Max drawdown < 5%
- Win rate > 55%
- No catastrophic failures in 2 weeks

**Initial Capital:** $10K (1 MNQ contract)

---

## Key Decisions

### **Why OFI Features?**
- **Academic proof:** 50+ papers show OFI predicts price moves (Cont 2014)
- **Industry validation:** Used by Citadel, Jane Street, Jump Trading
- **Our hypothesis:** If Sortino doesn't improve 1.0 to 2.0+, issue is RL setup (rewards/actions), not features

### **Why Mamba over LSTM?**
- **Better temporal modeling:** SSM architecture handles long sequences
- **Faster inference:** O(n) vs O(n²) for Transformers
- **Proven for time-series:** State-of-art for financial data

### **Why SAC over PPO?**
- **Continuous actions:** Better for position sizing
- **Sample efficient:** Learns from off-policy data
- **Stable:** Less prone to catastrophic forgetting

---

## Tech Stack

- **Data:** Databento (MBP-10 tick data)
- **Training:** SageMaker (ml.g4dn.xlarge spot GPU)
- **Framework:** Stable-Baselines3 + Mamba-SSM
- **Features:** NumPy, Pandas, HMMlearn
- **Config:** Hydra + OmegaConf
- **Infra:** Docker, AWS S3, CloudWatch

---

## Current Metrics (Phase 0)

- **Training data:** 97K bars (5 regimes, 2020-2023)
- **Validation:** 17K bars (Q1 2022)
- **Test:** 17K bars (Q1 2024)
- **Features:** 6 (lagging indicators)
- **Status:** Smoke test queued

---

## Next Steps

1. Run smoke test on `ml.g4dn.xlarge`
2. Measure baseline Sortino
3. Order Databento MBP-10 data
4. Build OFI preprocessing pipeline
5. Implement Phase 2 features

# V2: Hybrid Regime-Aware Trajectory Prediction

## Core Insight

O problema atual é que o **Fluid Ball/Lorenz** está sendo usado como um **modelo global**, enquanto o jogo de futebol é um sistema **híbrido**:

- **Regiões lineares/contínuas:** jogadores correndo, bola em posse, deslocamentos suaves.
- **Regiões dinâmicas/discretas:** passe, chute, mudança brusca de direção, disputa pela bola.

Em vez de `Predição → Lorenz → Resultado`, ter:

```
Predição
     │
     ▼
Detector de regime
     │
 ┌───┴──────────┐
 │              │
 ▼              ▼
Regime suave   Regime dinâmico
(linear)       (Lorenz / não linear)
 │              │
 └──────┬───────┘
        ▼
Predição final
```

- **Modelo linear** quando: velocidade baixa, aceleração pequena, direção estável.
- **Modelo Lorenz (ou outro não linear)** quando: aceleração cresce, mudança de direção grande, forte interação bola-jogadores.

### Por que isso faz sentido

Os experimentos mostraram que:

- γ=0.6 funciona porque estabiliza o sistema.
- O problema não é falta de caos; é que o caos aparece apenas em alguns momentos.
- Aplicar um modelo caótico continuamente pode "inventar" dinâmica onde ela não existe, enquanto amortecer continuamente pode "matar" dinâmica onde ela deveria existir.

Portanto, um **modelo híbrido** é mais adequado do que um Lorenz sempre ativo.

## Proposta para V2

Dividir a dinâmica em dois níveis:

1. **Nível local (90% do tempo):**
   - dinâmica linear ou quase linear;
   - filtro físico simples;
   - estabilidade.

2. **Nível de eventos (10% do tempo):**
   - Lorenz, sistema não linear ou outro modelo de alta dinâmica;
   - ativado apenas quando um detector indicar mudança de regime.

Isso modela o jogo como um **sistema híbrido**, alternando entre dinâmica contínua e dinâmica não linear.

## Implementation Options

### A. Confidence-based γ (simplest)

Replace `γ = 0.6` constant with `γ = f(confidence)`:

```
γ = γ_max * c + γ_min * (1 - c)
```

where `c ∈ [0,1]` is a confidence estimate that could come from:
- Ensemble variance (multiple forward passes with dropout)
- Ball velocity stability (cosine similarity across recent frames)
- Acceleration magnitude (high accel = low confidence)
- Learned predictor head

### B. Temporal γ (already implemented)

`--fluid_ball_gamma_target` and `--fluid_ball_gamma_tau` parameters added to the codebase.
Decays γ from `--fluid_ball_gamma` to `--fluid_ball_gamma_target` with time constant τ.

**Result**: time decay alone does NOT help because ball predictions don't improve over time — they diverge. Confident γ needs a **measured** signal, not a scheduled one.

### C. Learned kick/event classifier (most powerful)

Train a binary classifier to detect regime changes:
- Input: velocity history (10-20 frames), acceleration, ball-player distance
- Feature: sudden changes in ball velocity magnitude/direction, proximity to players
- Output: `p(regime_change)` or `p(kick_event)`
- Target: `.rcg` event logs (kicks, tackles) for supervised training
- When event detected: reset ball velocity to network prediction, lower γ
- When smooth: keep high γ for stability

### D. Hybrid rollout

```
if p(regime_change) > threshold:
    # event mode: trust network prediction, low damping
    ball_vel = network_prediction * (1 - γ_low * dt) + noise
    γ_low = 0.25
else:
    # smooth mode: stabilize, high damping
    ball_vel = network_prediction * (1 - γ_high * dt) + noise
    γ_high = 0.6
```

Or continuous blending:

```
γ = γ_high * (1 - p_change) + γ_low * p_change
```

## Key Lessons from V1

1. **Intercept correction is responsible for ~60% of gain** — keep it in V2
2. **Constant γ=0.6 beats time-decay γ** because ball predictions never become reliable over 30 frames
3. **Ball is only 5-19% of error metric** (1 of 12 agents) — improving ball alone gives limited gain
4. **All 4 official scenes are BALL-AT-FEET** — no kicks at boundary, so the model never saw an event
5. **The intercept needs a stable ball target** — high γ keeps ball in small region, making intercept consistent

## Dados Disponíveis para Treino

- `.rcg.gz` files: full match logs with ball/player positions
- `.rcl.gz` files: play-by-play event logs (kicks, fouls, goals, etc.)
- Event CSVs: parsed events in `ground-truth/`
- Challenge has official training data with frame-by-frame tracking

O `.rcl` contém eventos como `kick`, `tackle`, `pass`, `goal` com timestamps exatos — ideais para treinar o detector de regime.

## Next Steps

1. Parse `.rcl` files to extract kick events with frame numbers
2. Extract ball velocity features from `.rcg` frames
3. Train a lightweight classifier (XGBoost or small MLP) on (features → kick/no-kick)
4. Integrate into particle_filter.py as `use_event_aware` mode
5. Evaluate on test_old first, then one-shot on 2026

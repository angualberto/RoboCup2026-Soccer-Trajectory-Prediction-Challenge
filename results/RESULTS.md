# STP Challenge 2026 — Resultados

## Submissão Final

**Heun (RK2) integrator + Fluid Ball γ=0.6 σ=0.02 + Intercept + PF**
- Erro médio 2026 oficial: **14.74m**
- Arquivo: `results/test/challenge_submission.zip`

## Ablation Table (2026 oficial)

| Config | Avg Error | Δ vs GTPA |
|--------|:---------:|:---------:|
| RNN | 39.35m | - |
| GTPA Baseline | 56.87m | - |
| +PF+RM+FB γ=0.4 | 15.46m | -72.8% |
| +PF+RM+FB γ=0.6 | 14.92m | -73.8% |
| **+ Heun integrator** | **14.74m** | **-74.1%** |
| — sem intercept | 32.36m | -43.1% |

**Ablation test_old (2025, 3 cenas)**

| Config | Avg Error |
|--------|:---------:|
| FB γ=0.4 (user's original) | 7.14m |
| FB γ=0.6 σ=0.02 | 6.95m |
| **+ Heun integrator** | **6.85m** |
| — sem intercept | 13.85m |

## Integrator Comparison

| Integrator | test_old | 2026 |
|------------|:--------:|:----:|
| legacy (Euler in network) | 6.95m | 14.92m |
| Euler (pos_t + v·dt) | 7.60m | — |
| **Heun (RK2)** | **6.85m** | **14.74m** |
| Simpson 1/3 | 6.96m | 14.74m |
| AB2 | 7.67m | — |

Heun = `x_{t+1} = x_t + (v_t + v_{t+1})/2 · dt`. Consulta a velocidade amortecida duas vezes e faz a média. Reduz erro de integração sem custo adicional relevante.

## Erro por Cena (2026)

| Cena | Match | Frames | γ=0.6 | Sem Intercept | Δ |
|:----:|-------|:-----:|:----:|:------------:|:-:|
| 1 | HELIOS2024_2 vs CYRUS_0 | 2798→2828 | 14.60m | 32.74m | +124% |
| 2 | HELIOS2024_1 vs CYRUS_0 | 2293→2323 | 17.23m | 30.15m | +75% |
| 3 | CYRUS_1 vs HELIOS2024_1 | 472→502 | 13.99m | 32.79m | +134% |
| 4 | CYRUS_1 vs HELIOS2024_0 | 1168→1198 | 13.85m | 33.74m | +144% |

## Diagnóstico

1. **Intercept correction é o componente mais importante** — responsável por ~60% do ganho total. Sem ele, todas as cenas dobram de erro. A hipótese de que ele atrapalhava em cenas longas foi REFUTADA.

2. **Fluid Ball γ=0.6** funciona como amortecedor de velocidade: em cenários "BALL AT FEET" (todas as 4 cenas de 2026), a rede GTPA não tem informação de kicks, então a velocidade inicial da bola no frame de boundary pode estar errada. γ alto mata essa velocidade mais rápido, deixando o intercept puxar os jogadores na direção correta.

3. **Ball = 5-19% do erro** (1/12 agentes). Dominância é dos left players (81-95%). Melhorar left player prediction via ball context dá mais retorno que melhorar a bola em si.

4. **Todas as 4 cenas são "BALL AT FEET"** — nenhum kick no boundary. O modelo não falha por excesso de intercept, mas por falta de predição de eventos.

## Próximo Salto (V2)

Modelagem explícita de mudança de regime:
- `play → pass → shot → drift`
- Detectar kicks/eventos no histórico
- Condicionar trajetória prevista ao regime detectado
- Intercept adaptativo por regime (ativar mais em `pass`, menos em `drift`)

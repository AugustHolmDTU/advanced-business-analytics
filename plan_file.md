# Disruption Milestone Plan

## Goal
This branch, `Saxe_disruptions`, is for disruption-only work on the queue simulator. The purpose is to stress the existing two-city corridor setup and make the effect of disruptions easy to read in local plots, CSV outputs, and W&B.

What stays unchanged in this phase:
- 2 cities
- 100 km straight road
- one charging station at the midpoint
- shared queue
- 5-minute timesteps
- multi-day simulation support

What is explicitly postponed:
- mobile charging stations
- heuristic control policies
- RL wrappers
- DQN training

## Implemented Disruption Types
The simulator should support exactly these v1 disruptions:
- `capacity_drop`: midpoint capacity drops from 10 plugs to a lower value
- `station_outage`: midpoint capacity becomes 0 for a time window
- `demand_surge_ab`: more traffic from City A to City B
- `demand_surge_ba`: more traffic from City B to City A
- `service_time_inflation`: longer charging sessions during the disruption window

Defaults for the disruption config:
- at most one disruption event per day
- at most one active disruption at a time
- disruptions are sampled independently per day in random mode
- normal days remain possible
- baseline demand stays unchanged outside disruption windows

## Config and Semantics
The disruption-capable config lives in:
- `configs/simulation/simple_corridor_queue_disruptions.yaml`

Important config fields:
- `simulation.disruption.enabled`
- `simulation.disruption.mode`
- `simulation.disruption.daily_event_probability`
- `simulation.disruption.event_types`
- `simulation.disruption.capacity_drop.*`
- `simulation.disruption.station_outage.*`
- `simulation.disruption.demand_surge.*`
- `simulation.disruption.service_time_inflation.*`
- `simulation.disruption.scripted_events`

Supported modes:
- `random`: sample at most one event per day using the configured event probabilities and defaults
- `scripted`: reproduce fixed scenarios using `disruption_type`, `day_index`, `start_hour`, `duration_hours`, and `severity`

Scripted severity semantics:
- `capacity_drop`: severity means effective plug count during the disruption
- `station_outage`: severity is ignored
- `demand_surge_ab` / `demand_surge_ba`: severity means demand multiplier
- `service_time_inflation`: severity means service-time multiplier

## Required Metrics and Plots
Per-step disruption metrics:
- `disruption_active`
- `disruption_type`
- `disruption_day_index`
- `disruption_start_hour`
- `disruption_end_hour`
- `disruption_remaining_minutes`
- `effective_num_plugs`
- `demand_multiplier_ab`
- `demand_multiplier_ba`
- `service_time_multiplier`

Smoothed operational metrics:
- `queue_length_rolling_1h`
- `utilization_rolling_1h`
- `arrivals_total_rolling_1h`
- `starts_total_rolling_1h`
- `completions_total_rolling_1h`
- `queue_wait_mean_minutes_rolling_1h`

Summary outputs should include:
- disruption counts by type
- total disrupted minutes
- per-type mean queue length during disruption
- per-type mean wait time during disruption
- non-disruption baseline metrics

Expected local plot artifacts:
- queue length over time with shaded disruption windows
- smoothed queue and wait profile
- expected traffic versus realized arrivals
- effective capacity over time
- per-day queue overlays with disruption labels

## Success Criteria
This milestone is successful when:
- a no-disruption 3-day run still looks like the current baseline
- a disruption-enabled 3-day run clearly shows stressed periods
- outage and capacity-drop windows visibly increase queue and wait time
- plots and W&B metrics make the causal link between disruption timing and queue behavior obvious

## Recommended Run Commands
Baseline:

```bash
PYTHONPATH=src python -m evch.train.run_simple_corridor_sim \
  --config configs/simulation/simple_corridor_queue.yaml \
  --config configs/logging/wandb_online.yaml \
  --config configs/experiment/simple_corridor_queue.yaml
```

Random disruptions:

```bash
PYTHONPATH=src python -m evch.train.run_simple_corridor_sim \
  --config configs/simulation/simple_corridor_queue_disruptions.yaml \
  --config configs/logging/wandb_online.yaml \
  --config configs/experiment/simple_corridor_queue.yaml
```

For scripted debugging, switch the disruption config to `mode: scripted` and edit `scripted_events`.

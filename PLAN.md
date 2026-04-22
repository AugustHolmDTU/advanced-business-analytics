# Plan File: Mobile Stations + Disruption-Aware DQN Roadmap

## Summary
Build the next phase on the new car-level queue simulator, not the older zone-based RL corridor env. The implementation should proceed in four steps: add visible disruptions to the simulator, add one mobile station with simple discrete movement, benchmark rule-based heuristics, then wrap the simulator as a DQN-friendly RL environment.

The plan file to create later should be named `plan_file.md` at the repo root. Its purpose is to be the single roadmap for the queue-simulation branch of the project.

## Key Implementation Changes
### 1. Extend the queue simulator with explicit disruptions
Add a disruption layer to the queue-simulator config and runtime model. The disruption schedule must be sampled per episode and logged per timestep so W&B can show when the environment is under stress.

Use these v1 disruption types:
- `fixed_station_capacity_drop`: reduce midpoint plugs from `10` to a configured lower number for a time window
- `fixed_station_outage`: midpoint station capacity becomes `0` for a time window
- `directional_demand_surge_ab` and `directional_demand_surge_ba`: multiply expected passing demand in one direction during a window
- `service_time_inflation`: increase mean charging duration during a window

Use these defaults:
- at most one active disruption type at a time in v1
- disruption windows are contiguous and sampled within a single day
- each day in a multi-day visualization run can independently be normal or disrupted
- RL training episodes should use `24` hours, while visualization runs can stay at `72` hours

Add per-step logged fields:
- `disruption_active`
- `disruption_type`
- `disruption_day_index`
- `disruption_remaining_minutes`
- `effective_fixed_capacity`
- `demand_multiplier_ab`
- `demand_multiplier_ba`
- `service_time_multiplier`

Add smoothed versions for the most important operational metrics:
- `queue_length_rolling_1h`
- `utilization_rolling_1h`
- `arrivals_total_rolling_1h`
- `starts_total_rolling_1h`
- `completions_total_rolling_1h`

The simulation runner should also emit a local plot that overlays queue length and disruption windows, plus a per-day view showing whether the queue spike aligns with the disruption.

### 2. Add one mobile station as a simple controllable asset
Model one mobile charging unit with exactly three possible locations:
- `city_a`
- `midpoint`
- `city_b`

Movement rules:
- one action per timestep
- actions are `stay`, `move_to_city_a`, `move_to_midpoint`, `move_to_city_b`
- if already in transit, the only valid action is effectively `stay`
- travel takes a configured number of timesteps based on origin and destination
- while in transit, the mobile unit contributes zero charging capacity
- when parked, it adds extra plugs at its current location

Use these defaults:
- mobile unit adds `4` plugs
- travel time is `6` timesteps (`30` minutes) between adjacent nodes and `12` timesteps (`60` minutes) between end cities
- shared queue remains in place for v1, but a vehicle can use whichever capacity is active at its current node abstraction
- operationally, treat the mobile unit as augmenting one of three service points rather than introducing full route-level rerouting logic

To keep v1 decision-complete and DQN-friendly, refactor the simulator into a small service-network abstraction:
- service nodes are `city_a`, `midpoint`, `city_b`
- demand from each direction first maps to a preferred service node based on simple rules
- default mapping:
  - vehicles from either direction prefer `midpoint`
  - if midpoint is unavailable or heavily queued, overflow can be assigned to the city-end node on the vehicle’s current side only if mobile capacity is present there
- do not add arbitrary re-routing to far nodes in v1

This means the simulator remains queue-first and interpretable, while still making mobile placement meaningful.

### 3. Add heuristic policies before RL
Implement baseline control policies for the mobile unit before any DQN training. These are required both as comparators and as debugging tools.

Add these heuristic policies:
- `hold_midpoint`: keep the mobile unit at midpoint at all times
- `reactive_queue_threshold`: if queue exceeds a threshold and a disruption is active, move toward the affected best fallback node; otherwise hold midpoint
- `directional_surge_follower`: if AB surge is active, favor `city_a`; if BA surge is active, favor `city_b`; otherwise hold midpoint
- `oracle_simple`: optional benchmark using direct access to active disruption type for upper-bound intuition

Use these evaluation outputs:
- mean queue length
- peak queue length
- mean wait time
- max wait time
- total charging completions
- mean utilization
- disruption-window performance split vs non-disruption-window performance split

The simulator runner should support a `policy` mode so heuristics can be evaluated over many seeds before RL exists.

### 4. Wrap the simulator as a DQN environment
Create a new RL environment specifically for the queue simulator rather than modifying the existing placement env. The environment should expose a discrete action space and compact numeric observation vector.

Observation vector should include:
- `hour_of_day_sin`
- `hour_of_day_cos`
- `day_progress`
- `queue_length`
- `queue_length_rolling_1h`
- `queue_wait_mean_minutes`
- `queue_wait_max_minutes`
- `utilization`
- `arrivals_total_rolling_1h`
- `starts_total_rolling_1h`
- `completions_total_rolling_1h`
- `disruption_active`
- one-hot `disruption_type`
- `effective_fixed_capacity`
- `demand_multiplier_ab`
- `demand_multiplier_ba`
- `service_time_multiplier`
- one-hot mobile location
- `mobile_in_transit`
- `mobile_remaining_travel_steps`

Action space:
- `0 = stay`
- `1 = move_to_city_a`
- `2 = move_to_midpoint`
- `3 = move_to_city_b`

Reward:
- primary penalty on queue and waiting
- secondary reward for completions
- explicit movement penalty to discourage thrashing
- invalid-action penalty if trying to issue a move while already in transit

Use this default reward shape:
- `reward = 0.5 * completions_total - 0.25 * queue_length - 0.02 * queue_wait_mean_minutes - 0.05 * move_started`
- no reward term for expected demand directly
- no sparse “success/failure” terminal reward in v1

Training defaults:
- one-day episode length for RL
- random daily disruption sampling per episode
- DQN only after heuristics are stable
- train against a mix of normal and disrupted days, not disruption-only days
- evaluate on fixed held-out seeds and fixed scenario packs

## Important Interfaces / Types
Add these config sections:
- `simulation.disruption`
- `simulation.mobile_unit`
- `simulation.policy`
- `rl_mobile_env` or equivalent config block for DQN-specific episode and reward settings

Add these main runtime concepts:
- `DisruptionEvent` with type, start step, end step, and effect parameters
- `MobileUnitState` with location, in-transit flag, destination, and remaining travel steps
- `MobileCorridorEnv` as the new Gym/Gymnasium-compatible RL env for the queue simulator path

Keep the existing simple corridor runner intact, but extend it so it can:
- run uncontrolled simulations
- run heuristic-controlled simulations
- export disruption-aware and mobile-aware metrics to W&B and CSV

Do not retrofit this into the existing `ChargingPlacementEnv`. Keep the old environment as a separate benchmark track.

## Test Plan
Add deterministic unit tests for:
- disruption scheduling repeats correctly across multi-day visualization runs
- each disruption type changes only the intended simulator variables
- mobile unit travel timing and arrival logic
- mobile capacity contributes only when parked
- queue spikes become larger under outage and capacity-drop scenarios
- heuristic policies reduce queue or wait time versus `hold_midpoint` in at least one disruption regime
- RL env observation shape, action validity, reset behavior, and episode termination
- reward responds in the correct direction when queue length or movement cost increases

Add acceptance scenario tests:
- normal day with no disruption keeps queue moderate and mobile unit mostly unnecessary
- midpoint outage causes visible queue growth unless the mobile unit is reallocated
- AB directional surge makes `city_a` support more useful than midpoint-only behavior
- repeated fixed-seed runs produce identical summaries

## Assumptions and Defaults
- The queue-simulator branch is now the primary path for resilience experiments.
- The first mobile-station version uses one discrete mobile unit and three locations only.
- Disruptions are intentionally stylized and interpretable, not data-calibrated.
- RL is a comparator, not something that must outperform heuristics.
- Visualization runs stay multi-day for interpretability; RL runs stay single-day for tractability.
- The plan file to create later should be `plan_file.md` in the repo root and should summarize this roadmap, chosen defaults, open future extensions, and the exact milestone order above.

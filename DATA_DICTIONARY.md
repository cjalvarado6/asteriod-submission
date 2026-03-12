# Data Dictionary

Feature reference for the Asteroid Auction Challenge training dataset. Each row represents one asteroid with ~95 measurement features and target variables available only in training data.

**Important**: Market conditions and data collection methods may differ between the training dataset and live competition sectors. The training data spans multiple time periods to help you assess model generalization.

**Data Format**: Training data is provided as a Parquet file (`training.parquet`) with explicit column types. The types listed below match exactly what your strategy will receive during competition.

---

## Geological / Physical Properties

Measured by remote sensing and, where available, surface probe data.

| Feature | Type | Description |
|---------|------|-------------|
| `mass` | float64 | Estimated mass in kilotonnes. |
| `density` | float64 | Bulk density in g/cm3. Ranges from porous rubble piles (~1.5) to solid metal bodies (~8.0). |
| `porosity` | float64 | Fraction of volume that is void space. High porosity combined with low density may indicate rubble pile structure. |
| `spectral_class` | str | Tholen taxonomy classification. Values: `"C-type"`, `"S-type"`, `"M-type"`, `"X-type"`. |
| `mineral_signature_iron` | float64 | Spectroscopic concentration estimate for iron (Fe), scaled 0-1. |
| `mineral_signature_nickel` | float64 | Spectroscopic concentration estimate for nickel (Ni), scaled 0-1. |
| `mineral_signature_cobalt` | float64 | Spectroscopic concentration estimate for cobalt (Co), scaled 0-1. |
| `mineral_signature_platinum` | float64 | Spectroscopic concentration estimate for platinum group metals (PGM), scaled 0-1. |
| `mineral_signature_rare_earth` | float64 | Spectroscopic concentration estimate for rare earth elements (REE), scaled 0-1. |
| `albedo` | float64 | Surface reflectivity measured via photometry. Related to surface composition and weathering. |
| `rotation_period` | float64 | Sidereal rotation period in hours. Fast rotators may complicate surface operations. |
| `surface_roughness` | float64 | Terrain roughness index (0=smooth, 1=extremely rough). Affects landing and drill placement. |
| `magnetic_field_strength` | float64 | Measured magnetic field intensity in arbitrary units. Most asteroids have negligible fields. |
| `thermal_inertia` | float64 | Thermal inertia in SI units. Indicates surface heat response characteristics. |
| `shape_elongation` | float64 | Ratio of longest to shortest axis (1.0=spherical, 3.0+=highly elongated). |
| `regolith_depth` | float64 | Estimated depth of surface regolith layer in meters. |
| `water_ice_fraction` | float64 | Detected fraction of water ice (0-1). Many asteroids have zero water ice. |
| `volatile_content` | float64 | Fraction of volatile compounds detected. Includes ices and trapped gases. |
| `structural_integrity` | float64 | Engineering assessment of structural soundness (0=critically fractured, 1=solid monolith). |
| `estimated_volume` | float64 | Derived: mass / density. |
| `surface_gravity` | float64 | Derived surface gravitational acceleration. |
| `escape_velocity` | float64 | Derived escape velocity from the surface. |
| `composition_heterogeneity` | float64 | Compositional variability across the body (0=uniform, 1=highly mixed). *Uniform composition can simplify extraction planning.* |
| `subsurface_anomaly_score` | float64 | Radar-derived score indicating subsurface structural anomalies. *Anomalies may indicate valuable deposits or hidden risks.* |
| `crystalline_fraction` | float64 | Fraction of mineral content in crystalline vs amorphous form. *Higher crystalline content may indicate more extractable ore.* |

---

## Orbital / Location Properties

Derived from ephemeris data and belt surveys.

| Feature | Type | Description |
|---------|------|-------------|
| `semi_major_axis` | float64 | Orbital semi-major axis in AU. |
| `eccentricity` | float64 | Orbital eccentricity. Most belt asteroids have near-circular orbits. |
| `inclination` | float64 | Orbital inclination in degrees relative to the ecliptic. |
| `delta_v` | float64 | Required velocity change (km/s) to reach the asteroid from the nearest transfer point. |
| `belt_region` | str | Categorical belt position. Values: `"inner"`, `"main"`, `"outer"`. |
| `cluster_id` | int64 | Geological cluster assignment. Asteroids in the same cluster share a common formation history. |
| `orbital_period` | float64 | Orbital period in years (derived from semi-major axis). |
| `perihelion_distance` | float64 | Closest approach to the Sun in AU. |
| `aphelion_distance` | float64 | Farthest distance from the Sun in AU. |
| `transfer_window_frequency` | float64 | How often optimal transfer windows occur (higher=more frequent). |
| `nearest_station_distance` | float64 | Distance to the nearest logistics station in AU. |
| `piracy_proximity_index` | float64 | Proximity to known piracy corridors. Higher values indicate greater security cost. |
| `communication_delay` | float64 | One-way light-time communication delay in minutes. |
| `orbital_stability_score` | float64 | Long-term orbital stability assessment (0=unstable, 1=highly stable). |
| `conjunction_frequency` | float64 | Rate of close approaches with other bodies (scaled). |
| `lucky_number` | float64 | Numerological favorability score (0-10). *Derived from orbital resonance patterns.* |

---

## Survey / Exploration Data

Compiled from prospecting missions. Survey methodology and timing vary.

| Feature | Type | Description |
|---------|------|-------------|
| `survey_confidence` | float64 | Overall confidence rating of the survey data (0-1). |
| `probe_type` | str | Survey probe deployed. Values: `"passive"`, `"active_flyby"`, `"landing"`, `"drill_core"`. |
| `surveyor_reputation` | float64 | Reputation score of the surveying firm (0-1). *Reputation scores are self-reported and updated annually.* |
| `num_surveys` | int64 | Number of independent survey missions conducted. |
| `conflicting_results` | int64 | Binary flag (0 or 1). Set to 1 if surveys produced contradictory findings. |
| `extraction_difficulty` | float64 | Engineering assessment of extraction difficulty (0=trivial, 1=extremely difficult). |
| `accessibility_score` | float64 | How accessible deposits are to current drilling technology (0-1). |
| `survey_age_years` | float64 | Time since the most recent survey in years. |
| `data_completeness` | float64 | Fraction of standard survey measurements successfully collected (0-1). |
| `spectral_resolution` | float64 | Resolution quality of the spectroscopic instruments used (0-1). |
| `ground_truth_samples` | int64 | Number of physical samples returned for laboratory analysis. |
| `estimated_extraction_cost` | float64 | Surveyor's estimate of total extraction cost in thousands of credits. *Estimates assume standard equipment configurations.* |
| `drilling_feasibility` | float64 | Engineering assessment of drilling viability (0=infeasible, 1=ideal). *Based on current drilling technology standards.* |
| `equipment_compatibility` | float64 | Compatibility score with standard mining equipment (0-1). |
| `estimated_yield_tonnes` | float64 | Surveyor's estimate of extractable material in tonnes. |
| `survey_anomaly_flag` | int64 | Binary flag. Set to 1 if the survey team flagged unusual readings. |
| `dual_phase_extraction` | int64 | Binary flag. Set to 1 if asteroid requires two-phase extraction. *Determined during survey based on detected ore composition.* |
| `previous_claim_history` | int64 | Number of times this asteroid has been previously claimed and abandoned. |
| `legal_encumbrance_score` | float64 | Degree of legal complications (0=clear, higher=more encumbered). |
| `environmental_hazard_rating` | float64 | Environmental risk assessment (0=benign, 1=severe). |
| `insurance_risk_class` | int64 | Insurance underwriter risk classification (1=lowest risk, 5=highest risk). |
| `ai_valuation_estimate` | float64 | Automated valuation from ValuCorp v2.3. *Use with caution.* |
| `analyst_consensus_estimate` | float64 | Median valuation from independent mining analysts. *Recent investigations have raised questions about analyst independence.* |

---

## Market / Economic Conditions

Snapshot of market conditions at the time of auction. Prices reflect current spot rates.

| Feature | Type | Description |
|---------|------|-------------|
| `mineral_price_iron` | float64 | Current spot price for iron per unit. |
| `mineral_price_nickel` | float64 | Current spot price for nickel per unit. |
| `mineral_price_cobalt` | float64 | Current spot price for cobalt per unit. |
| `mineral_price_platinum` | float64 | Current spot price for platinum group metals per unit. |
| `mineral_price_rare_earth` | float64 | Current spot price for rare earth elements per unit. |
| `mineral_price_water` | float64 | Current spot price for water ice per unit. |
| `fuel_cost_per_unit` | float64 | Current fuel cost for transport vessels. |
| `insurance_rate` | float64 | Current insurance premium rate for mining operations. |
| `tax_rate` | float64 | Applicable extraction tax rate for this claim's jurisdiction. |
| `economic_cycle_indicator` | float64 | Macro-economic cycle position (below 1.0=contraction, above 1.0=expansion). |
| `market_volatility_index` | float64 | Current market volatility measure. |
| `demand_backlog_months` | float64 | Months of unfilled demand orders in the minerals market. |
| `shipping_congestion_factor` | float64 | Shipping lane congestion level (0=clear, 1=severely congested). |
| `refinery_capacity_utilization` | float64 | Fraction of system-wide refinery capacity in use. |
| `spot_vs_contract_spread` | float64 | Spread between spot and long-term contract prices. |
| `credit_availability_index` | float64 | Availability of financing (0=tight, 1=abundant). |
| `competitor_activity_level` | float64 | Estimated activity level of other mining operations in the region. |
| `regulatory_burden_score` | float64 | Regulatory overhead for operations in this jurisdiction. |
| `supply_chain_disruption_risk` | float64 | Assessed risk of supply chain disruptions. |
| `technology_readiness_level` | float64 | Technology readiness level of available equipment (scale 5-9). |
| `media_hype_score` | float64 | Composite media attention index (0-10+). *May not correlate with actual value.* |
| `social_sentiment_score` | float64 | Aggregated sentiment from mining industry social feeds. *Notoriously noisy.* |

---

## Sector / Environmental Conditions

Local space environment near the asteroid.

| Feature | Type | Description |
|---------|------|-------------|
| `radiation_level` | float64 | Ambient radiation level in the asteroid's vicinity. |
| `micrometeorite_density` | float64 | Local micrometeorite flux density. |
| `solar_flux` | float64 | Solar energy flux at the asteroid's location (relative to Earth=1.0). |
| `infrastructure_proximity` | float64 | Proximity to existing mining infrastructure (0=remote, 1=well-connected). *Measured at time of survey; infrastructure expands over time.* |
| `navigation_complexity` | float64 | Complexity of navigation in the local orbital environment. |
| `rescue_response_time_hours` | float64 | Estimated emergency response time from nearest rescue facility. |
| `local_jurisdiction_stability` | float64 | Political stability of the governing jurisdiction (0=unstable, 1=stable). |
| `worker_availability_index` | float64 | Availability of qualified mining crews in the region. |
| `power_grid_access` | float64 | Access to orbital power grid infrastructure. |
| `debris_field_density` | float64 | Density of debris in the local orbital environment. |

---

## Target Variables

Available in training data only. Not available during competition.

| Feature | Type | Description |
|---------|------|-------------|
| `mineral_value` | float64 | Total mineral content value. **Set to 0 for catastrophe or impacted rows.** |
| `extraction_yield` | float64 | Operational recovery factor (0-1+). **Set to 0 for catastrophe or impacted rows.** |
| `extraction_delay` | int64 | Extraction timeline in rounds until revenue arrives. |
| `catastrophe_type` | str | Catastrophe outcome. Values: `"none"`, `"void_rock"`, `"structural_collapse"`, `"toxic_outgassing"`. |
| `toxic_outgassing_impact` | int64 | Binary (0 or 1). Set to 1 if impacted by toxic outgassing from another asteroid in the same cluster. |

**Important**: Rows with `catastrophe_type != "none"` or `toxic_outgassing_impact == 1` have zeroed regression targets. When training models for `mineral_value` or `extraction_yield`, filter these rows. The catastrophe and impact columns allow you to learn the probability of these events from features.

---

## Metadata Columns

| Feature | Type | Description |
|---------|------|-------------|
| `asteroid_id` | str | Unique identifier for each asteroid. |
| `time_period` | str | Data collection period identifier (e.g., `"2045-Q1"`). **Do not use as a model feature.** |

---

## Catastrophe Risk Factors

Catastrophe probability is driven by observable features. Key risk indicators:

| Risk Factor | Description |
|-------------|-------------|
| **Structural Integrity** | Primary driver. Low integrity significantly increases risk. |
| **Density** | Low density may suggest hollow structures prone to collapse. |
| **Porosity** | High porosity combined with low density indicates fragile rubble pile structure. |
| **Volatile Content** | High volatile content increases toxic outgassing risk. |
| **Survey Confidence** | Low confidence means risks are harder to assess. |

Catastrophe rates vary significantly based on feature combinations. Asteroids with multiple risk factors are considerably more dangerous than those with strong structural indicators

## Catastrophe Penalties

When a catastrophe occurs, you lose your bid plus a flat penalty per catastrophe type:

| Catastrophe Type | Penalty |
|------------------|---------|
| **Void Rock** | $100 |
| **Structural Collapse** | $200 |
| **Toxic Outgassing** | $300 + $10 × (other asteroids in cluster) |

**Toxic outgassing** is particularly dangerous: if ANY asteroid in a cluster experiences toxic outgassing, ALL other asteroids in that cluster have their extraction yield reduced to zero. This is reflected in the training data via the `toxic_outgassing_impact` column.

---

## Notes

- Not all features are equally informative. Part of the challenge is determining which measurements carry signal.
- Market conditions in the training data reflect one economic period. Competition sectors may differ.
- Some features are derived from or correlated with others.
- Asteroid valuation involves interacting factors. Simple univariate relationships may not capture the full picture.
- The `economic_cycle_indicator` feature is consistent for all asteroids in a given round and reflects the current market regime (0.7 = bust, 1.0 = normal, 1.4 = boom).

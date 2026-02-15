# Weather Pattern Movement Around NYC and Temperature Prediction from Surrounding Stations

## Qualitative Research Background for the NYC Temperature Prediction Project

---

## 1. How Weather Moves Around New York City

### Prevailing Winds

New York City's weather is governed by the mid-latitude westerlies — the broad planetary-scale wind belt that carries weather systems generally from west to east across the continental United States. Within this overall pattern, NYC experiences significant seasonal variation in wind direction.

In **summer**, winds are predominantly from the **southwest**. This flow transports warm, humid air that has been conditioned over the Gulf of Mexico and the subtropical Atlantic northward into the Northeast. These southwesterly winds are the reason NYC summers feature periods of high heat and humidity — the air mass arriving at NYC has been warmed over thousands of miles of land and subtropical water.

In **winter**, winds shift to predominantly **northwesterly**. This brings cold, dry continental air masses that originate over Canada and the Arctic interior. These polar and arctic air masses are the source of NYC's coldest days. A strong cold front followed by sustained northwesterly winds can drop temperatures dramatically in a matter of hours — a textbook example of cold-air advection.

During **transitional seasons** (spring and fall), wind direction is more variable, alternating between these regimes as competing air masses battle for dominance. This variability is one reason spring and fall temperature forecasting is harder than summer or winter.

### The Three Dominant Air Masses

NYC sits at the intersection of three major air mass source regions, and the character of any given day's weather depends on which air mass is in control:

1. **Continental Polar / Arctic (cP/cA):** Cold, dry air from northern Canada. Arrives on northwest winds, especially in winter. Produces the coldest days and is associated with post-frontal clear skies.

2. **Maritime Tropical (mT):** Warm, humid air from the Gulf of Mexico and subtropical Atlantic. Arrives on southwest to south winds, dominant in summer. Produces the warmest, most humid conditions.

3. **Maritime Polar (mP):** Cool, damp air from the North Atlantic. Arrives on northeast to east winds. Associated with nor'easters and overcast, raw conditions. This air mass is unique to the coastal Northeast and produces some of NYC's most unpleasant weather — chilly, gray, and damp.

### Storm Tracks and Frontal Passages

Mid-latitude cyclones (low-pressure systems) are the primary mechanism for day-to-day weather changes in NYC. These systems follow several characteristic tracks:

**The Alberta Clipper track:** Fast-moving systems that originate over Alberta, Canada and sweep southeast across the Great Lakes and into the Northeast. These bring brief bursts of snow or cold rain followed by sharp cold-air advection from the northwest. Because they move quickly, temperatures at stations to the west and northwest of NYC (Scranton, Albany, Binghamton) will show the cold-air arrival 12–24 hours before NYC does.

**The Gulf Coast / East Coast track (Nor'easters):** Low-pressure systems that form or intensify along the Gulf Coast or the Carolinas and ride northward along the Eastern Seaboard. These draw warm, moist air northward ahead of them (warm advection from the south and southwest) and cold air behind them (cold advection from the northwest). Stations to the south and southwest of NYC (Philadelphia, Trenton, Atlantic City) often feel the warm-air surge first, while stations to the north and west see the cold air arrive earlier.

**The Ohio Valley track:** Systems that move eastward through the Ohio Valley and into the mid-Atlantic. These are the most common cyclone track affecting NYC and produce classical warm-front-then-cold-front sequences. Temperatures at stations to the west (Allentown, Scranton) and southwest (Philadelphia) provide a 12–24 hour preview of what NYC will experience.

### Coastal and Local Effects

NYC's position on the coast introduces additional complexity. The Atlantic Ocean acts as a thermal moderator — warming the city in winter relative to inland areas and cooling it in summer, particularly when onshore (easterly) winds blow. This marine influence means that the relationship between NYC's temperature and inland stations is not purely a function of advection; it also depends on wind direction. When winds are offshore (westerly), NYC's temperature tracks inland stations closely. When winds are onshore (easterly), the ocean decouples NYC from the interior.

The urban heat island effect also plays a role. Central Park — the official NYC observation station — is warmer than surrounding suburban and rural stations, particularly overnight. This systematic bias is relatively stable and is something a neural network should learn from the data.

---

## 2. Temperature Advection: The Physics Behind the Prediction

### What Is Temperature Advection?

Temperature advection is the horizontal transport of temperature by the wind. It is the physical process that makes predicting a city's temperature from upwind stations scientifically grounded, not just a statistical trick.

The advection equation states that the rate of temperature change at a fixed point is proportional to the wind speed and the temperature gradient in the direction the wind is blowing from. In simplified form:

```
ΔT/Δt ≈ -V × (ΔT/Δd) × cos(θ)
```

Where V is wind speed, ΔT/Δd is the temperature gradient between upwind and downwind stations, and θ is the angle between the wind direction and the temperature gradient. This equation tells us that if there is a strong wind blowing from a region that is significantly colder (or warmer) than the forecast location, the temperature at the forecast location will change accordingly.

**Warm-air advection (WAA)** occurs when the wind blows from warmer regions toward cooler ones. The forecast location warms. This commonly occurs ahead of warm fronts, with southerly or southwesterly winds pushing Gulf-warmed air into the Northeast.

**Cold-air advection (CAA)** occurs when the wind blows from colder regions toward warmer ones. The forecast location cools. This commonly occurs behind cold fronts, with northwesterly winds driving Arctic air southeastward.

### Why This Matters for the Neural Network

Temperature advection is the primary reason that yesterday's temperatures at surrounding stations contain predictive information about today's temperature in NYC. When a cold air mass is being advected southeastward, stations to the northwest of NYC (Albany, Poughkeepsie, Scranton) will record colder temperatures on day t−1, foreshadowing the cold air's arrival at NYC on day t. Similarly, warm air being advected northeastward will show up first at stations to the southwest (Philadelphia, Trenton).

Critically, temperature advection is not the only process controlling NYC's temperature. Other factors include solar radiation (diurnal heating), cloud cover, precipitation (evaporative cooling), snow cover (albedo), and local effects. However, in the mid-latitudes during winter, advection is often the dominant factor, and even in summer it remains an important contributor. This is why the neural network approach has a strong physical basis: it is learning the advection signal embedded in the station network.

### Which Directions Matter Most?

Given NYC's position and the prevailing circulation patterns, the stations likely to carry the strongest predictive signal are:

- **West and northwest stations** (Allentown, Scranton, Albany): Most weather approaches NYC from this direction in the prevailing westerlies. These stations are the "upstream" sensors for the dominant flow pattern.
- **Southwest stations** (Philadelphia, Trenton): Important for detecting warm-air advection ahead of warm fronts and ahead of approaching cyclones.
- **South and southeast stations** (Atlantic City, coastal NJ): Important for detecting maritime influences and onshore flow events.
- **North and northeast stations** (Hartford, Bridgeport): Important for detecting cold air damming and northeast flow events (nor'easters).
- **Nearby stations** (Newark, LaGuardia, JFK, White Plains): Capture the local thermal environment and provide near-field spatial context. These will have very high same-day correlation with Central Park but may have less predictive lead time.

A well-designed neural network should assign larger weights to the west-northwest stations on average, reflecting the dominant advection direction, but should also learn to leverage other stations when the flow pattern deviates from the prevailing westerlies.

---

## 3. Prior Research: Neural Networks for Spatial Temperature Interpolation

### The Foundational Study: Snell, Gopal & Kaufmann (2000)

The most directly relevant academic work is Snell, Gopal, and Kaufmann's 2000 paper in the *Journal of Climate*, titled "Spatial Interpolation of Surface Air Temperatures Using Artificial Neural Networks: Evaluating Their Use for Downscaling GCMs."

This study used ANNs to estimate daily maximum surface air temperature (TMAX) at 11 interior locations using temperature data from a lattice of 16 surrounding stations. Key findings:

- The ANN outperformed traditional spatial interpolation methods (spatial average, nearest neighbor, inverse distance weighting) in 94% of comparisons on out-of-sample test data.
- The ANN "encompassed" the benchmark methods in 77% of comparisons, meaning the ANN's predictions contained all the useful information present in the benchmark predictions, plus additional signal.
- The 16-point ANN (using all surrounding stations) outperformed the 4-point ANN (using only the four nearest stations), demonstrating that more distant stations contribute useful information.
- Benchmark methods encompassed the ANN in only 2% of comparisons, confirming the ANN's informational superiority.

This study provides strong evidence that the neural network approach proposed in this project has a sound basis. It also suggests that using a larger lattice of surrounding stations is better than using only the nearest few.

### CNN-LSTM for Spatiotemporal Temperature Prediction

A study published in *Applied Artificial Intelligence* (2023) constructed a CNN-LSTM architecture to extract spatiotemporal features of temperature from multiple meteorological stations simultaneously. The CNN layers captured spatial correlations between stations, while the LSTM layers captured temporal dependencies. This approach outperformed standalone CNN and LSTM models and represents the state-of-the-art for problems like the one proposed in this project.

### Machine Learning vs. Traditional Interpolation

A comparative study in *Environmental Modeling & Assessment* (2023) evaluated three approaches for mapping urban air temperature: Ordinary Kriging (a geostatistical method), a statistical machine learning model, and a physics-based weather simulation model (WRF). The ML approach achieved an RMSE of 1.23°C and R² of 0.93, outperforming both the WRF model (RMSE 1.7°C) and Ordinary Kriging (RMSE 3.00°C).

### Kriging with External Drift

A 2024 study on spatial interpolation for agricultural decision support found that kriging with elevation as an external drift variable reduced mean absolute error for hourly air temperature from 0.93°C (using the nearest station alone) to 0.59°C. This demonstrates the baseline performance one can achieve with traditional geostatistics — and sets a floor that neural network approaches should beat.

### Key Insight from the Literature

A consistent finding across the literature is that neural networks outperform traditional interpolation methods because they can capture **nonlinear relationships** and **combinative effects** between stations. Temperature fields are not simple smooth surfaces — they are disrupted by fronts, elevation changes, urban heat islands, and coastal effects. Linear methods like inverse distance weighting assume smooth, stationary spatial fields and cannot adapt to these complexities. Neural networks can.

---

## 4. Implications for This Project

### Station Selection Should Follow the Physics

The surrounding stations should be selected not just by proximity but by their **position relative to dominant advection pathways**. West-to-northwest stations should be prioritized because they are upstream in the prevailing flow. But south-southwest stations (warm advection pathway) and northeast stations (nor'easter pathway) should also be included. A ring of stations at varying distances captures both near-field persistence and far-field advection signals.

### The t−1 Lag Captures the Advection Timescale

Weather systems in the mid-latitudes typically move at 20–40 mph. A station 100–200 miles upwind of NYC will feel the effects of an approaching air mass roughly 3–10 hours before NYC does. Using day t−1 data (a ~24-hour lag) is well-matched to this advection timescale for stations in the 50–200 mile range. This is why the t−1 lag is a physically reasonable choice for the input window.

### Expect Seasonal Variation in Model Performance

The model will likely perform better in winter than in summer. In winter, temperature advection is the dominant process (weak solar heating, strong temperature gradients, active storm track), so the surrounding-station signal is strong. In summer, local convective processes, sea breezes, and solar heating play larger roles, which are less predictable from surrounding-station temperatures alone. The sensitivity analysis should examine seasonal MAE differences and consider whether season-specific models improve performance.

### The Autoregressive Term Will Be Powerful but Informative to Isolate

NYC's own TMAX at t−1 is a strong predictor of TMAX at t simply due to atmospheric persistence — weather changes gradually. Including this as an input will likely reduce MAE significantly. However, it is important to also evaluate the model **without** this term to understand how much predictive value the surrounding stations provide on their own. The surrounding-station-only model tests the core hypothesis about spatial temperature advection, while the combined model is the practical production version.

### The Neural Network Is Learning a Physical Process

This is not mere curve-fitting. The weights the network learns correspond to a real physical phenomenon: the differential importance of various upwind stations in predicting how temperature will change at NYC. If the learned weights are largest for the west-northwest stations (consistent with prevailing westerlies), this confirms the network is capturing the advection signal. Analyzing the learned weights provides both scientific insight and a validation check.

---

## 5. References

- Snell, S.E., S. Gopal, and R.K. Kaufmann, 2000: "Spatial Interpolation of Surface Air Temperatures Using Artificial Neural Networks: Evaluating Their Use for Downscaling GCMs." *Journal of Climate*, 13(5), 886–895.
- NOAA Climate of New York State, climatological summary. Available at geo.hunter.cuny.edu.
- Penn State METEO 3, "Temperature Advection," e-education.psu.edu/meteo3/l3_p7.html.
- University of Wisconsin–Madison, AOS, "Estimating Warm or Cold Air Advection," aos.wisc.edu.
- Mount Washington Observatory, "Terms Used in Forecasting: Advection," 2021.
- Huang et al., 2024: "A Spatial Interpolation Method for Meteorological Data Based on a Hybrid Kriging and Machine Learning Approach." *International Journal of Climatology*.
- CNN-LSTM spatial simulation, 2023: "Spatial Simulation and Prediction of Air Temperature Based on CNN-LSTM." *Applied Artificial Intelligence*.
- Interpolation comparison study, 2023: "Interpolation, Satellite-Based Machine Learning, or Meteorological Simulation?" *Environmental Modeling & Assessment*.
- Agricultural DSS interpolation study, 2024: "Near real-time spatial interpolation of hourly air temperature and humidity for agricultural decision support systems." *Computers and Electronics in Agriculture*.

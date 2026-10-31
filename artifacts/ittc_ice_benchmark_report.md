# ITTC Ice Pool Test Benchmarking Report

**Generated:** 2026-10-31 04:18:24 UTC
**Standard:** ITTC 7.5-04-01-01.1
**Mission ID:** 718

## Acceptance Criteria

| Criterion | Value | Threshold | Status |
|-----------|-------|-----------|--------|
| Ship types validated | 4 | >= 3 | ✅ PASS |
| Auto report generated | True | True | ✅ PASS |
| Reproducibility (CV) | 0.0000 | < 0.02 | ✅ PASS |

## Ice Properties

| Property | Value | Unit |
|----------|-------|------|
| Thickness | 0.80 | m |
| Flexural strength | 500 | kPa |
| Crushing strength | 3.0 | MPa |
| Density | 900 | kg/m³ |
| Friction coefficient | 0.050 | - |

## Ship Results

### Container (Arctic Container 4400)

**Ship Geometry:**

- L_pp = 230.0 m, B = 32.2 m, T = 10.5 m
- Cb = 0.655, Cwp = 0.785
- Bow angle = 22.5°, Stem angle = 15.0°
- Ice class: 1A

**Ice Resistance Comparison:**

| Quantity | Predicted | Reference | Error |
|----------|-----------|-----------|-------|
| Total resistance | 3115.1 kN | 2850.0 kN | 9.3% |

**Resistance Breakdown:**

- Ice breaking: 2128.1 kN
- Ice clearance: 68.9 kN
- Open water: 904.4 kN
- Ice friction: 13.6 kN

**Status:** ✅ PASS (error within 15% tolerance)

### Bulk (Baltic Bulk 50000)

**Ship Geometry:**

- L_pp = 190.0 m, B = 32.0 m, T = 11.2 m
- Cb = 0.820, Cwp = 0.830
- Bow angle = 25.0°, Stem angle = 18.0°
- Ice class: 1A

**Ice Resistance Comparison:**

| Quantity | Predicted | Reference | Error |
|----------|-----------|-----------|-------|
| Total resistance | 3454.6 kN | 3200.0 kN | 8.0% |

**Resistance Breakdown:**

- Ice breaking: 2639.2 kN
- Ice clearance: 68.4 kN
- Open water: 736.7 kN
- Ice friction: 10.3 kN

**Status:** ✅ PASS (error within 15% tolerance)

### Research (Polar Research 3000)

**Ship Geometry:**

- L_pp = 85.0 m, B = 18.5 m, T = 6.2 m
- Cb = 0.580, Cwp = 0.760
- Bow angle = 18.0°, Stem angle = 12.0°
- Ice class: PC6

**Ice Resistance Comparison:**

| Quantity | Predicted | Reference | Error |
|----------|-----------|-----------|-------|
| Total resistance | 736.6 kN | 680.0 kN | 8.3% |

**Resistance Breakdown:**

- Ice breaking: 580.7 kN
- Ice clearance: 9.1 kN
- Open water: 144.3 kN
- Ice friction: 2.5 kN

**Status:** ✅ PASS (error within 15% tolerance)

### Tanker (Arctic Product Tanker 30000)

**Ship Geometry:**

- L_pp = 183.0 m, B = 27.4 m, T = 9.8 m
- Cb = 0.780, Cwp = 0.810
- Bow angle = 21.0°, Stem angle = 14.0°
- Ice class: 1AS

**Ice Resistance Comparison:**

| Quantity | Predicted | Reference | Error |
|----------|-----------|-----------|-------|
| Total resistance | 2666.4 kN | 2450.0 kN | 8.8% |

**Resistance Breakdown:**

- Ice breaking: 1897.5 kN
- Ice clearance: 51.8 kN
- Open water: 706.8 kN
- Ice friction: 10.4 kN

**Status:** ✅ PASS (error within 15% tolerance)

## Summary

- **Ships tested:** 4
- **Ships within tolerance:** 4
- **Mean relative error:** 8.6%
- **Max relative error:** 9.3%
- **Reproducibility (CV):** 0.0000

**Overall Assessment:** ✅ ALL CRITERIA PASS

## Audit Trail

- Framework version: 1.0.0
- Model: ITTC_Lindqvist_Riska_combined
- Reproducibility runs: 5
- container_vessel: hash=12d0585bad6ad98c, timestamp=2026-10-31T04:18:24Z
- bulk_carrier: hash=a8559259e2e93cf9, timestamp=2026-10-31T04:18:24Z
- research_vessel: hash=17f00bce1df544b2, timestamp=2026-10-31T04:18:24Z
- tanker: hash=93f759a142f1135b, timestamp=2026-10-31T04:18:24Z

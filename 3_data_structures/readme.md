

## ✅ **Lists — store multiple objects (stars, pulsars, planets)**

### 📌 Code

```python
# list of pulsars observed
pulsars = ["Crab", "Vela", "Geminga"]

print("First pulsar:", pulsars[0])

# add new detected pulsar
pulsars.append("PSR B1937+21")

# remove a pulsar
pulsars.remove("Vela")

# update a name (example correction)
pulsars[1] = "Geminga Pulsar"

# list comprehension: convert names to uppercase
upper_pulsars = [p.upper() for p in pulsars]
print("Uppercase:", upper_pulsars)

# iterate
for p in pulsars:
    print("Observed:", p)
```

### 🛰️ Notes (Markdown)

| Concept    | Astronomy Meaning                       |
| ---------- | --------------------------------------- |
| Ordered    | sequence of targets in observation list |
| Mutable    | you can add new sky objects             |
| Duplicates | allowed (multiple exposures)            |

Good for: **target catalogues**, observation sequences, RA/DEC lists

---

## ✅ **Tuples — fixed celestial coordinates or immutable catalog entries**

### 📌 Code

```python
# RA and DEC of Crab Pulsar (in degrees)
crab_coords = (83.6331, 22.0145)

ra, dec = crab_coords
print("RA:", ra, "DEC:", dec)

# tuple of spectral bands observed
bands = ("X-ray", "Radio", "Gamma-ray")
print("Bands observed:", bands)

# single-element tuple for catalog ID
catalog_id = ("PSR J0534+2200",)
print(type(catalog_id))
```

### 🛰️ Notes

Use when values shouldn't change
Examples: **coordinates, catalog IDs, instrument settings**

---

## ✅ **Sets — remove duplicates (unique stars, detectors, filters)**

### 📌 Code

```python
# repeated stars in multiple observation runs
stars = {"Sirius", "Vega", "Sirius", "Betelgeuse"}

print("Unique targets:", stars)

# add new target
stars.add("Rigel")

# detector names on a telescope
detectors_A = {"UV", "Optical", "IR"}
detectors_B = {"IR", "Gamma", "X-ray"}

print("Union:", detectors_A | detectors_B)        # all bands
print("Intersection:", detectors_A & detectors_B) # shared band
print("Unique to A:", detectors_A - detectors_B)
```

### 🛰️ Notes

Used for **unique celestial objects**, **instrument lists**, **observation filters**

---

## ✅ **Dictionaries — store object properties & astronomy metadata**

### 📌 Code

```python
# info about a pulsar
crab = {
    "name": "Crab Pulsar",
    "period_ms": 33.0,
    "distance_ly": 6500,
    "wavelengths": ["X-ray", "Radio", "Optical"]
}

print(crab["name"])
print("Period (ms):", crab.get("period_ms"))

# add flux measurement
crab["flux_mJy"] = 14.5

# iterate through data
for key, val in crab.items():
    print(key, ":", val)

# remove a field
crab.pop("flux_mJy")
```

### 🛰️ Notes

Great for **FITS header info**, **object catalogs**, **metadata**, JSON astronomic APIs

---

## ✅ Quick Astro-Comparison Table

| Type  | Example Astro Use                               |
| ----- | ----------------------------------------------- |
| List  | Target list, exposure times, RA array           |
| Tuple | Fixed RA/DEC pair, pixel coordinates            |
| Set   | Unique star IDs in survey, filter wheels        |
| Dict  | Object metadata, FITS header, instrument config |

---


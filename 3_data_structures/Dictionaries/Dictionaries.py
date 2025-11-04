
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

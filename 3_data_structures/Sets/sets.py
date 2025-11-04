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

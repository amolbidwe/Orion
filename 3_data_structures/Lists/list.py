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

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

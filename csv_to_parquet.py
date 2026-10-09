import polars as pl

    
neurons = pl.read_csv("data/neurons.csv")
connections = pl.read_csv("data/connections_princeton.csv")

# Create lookup table from neurons
nt_lookup = (
    neurons
    .select([
        pl.col("Root ID"),
        pl.col("Predicted NT type").alias("nt_type")
    ])
)

# Populate nt_type using the presynaptic neuron
connections = (
    connections
    .drop("nt_type", strict=False)  # Remove empty column if it exists
    .join(
        nt_lookup,
        left_on="pre_root_id",
        right_on="Root ID",
        how="left",
    )
)

# Optional: report any unmatched neurons
missing = connections.filter(pl.col("nt_type").is_null()).height
print(f"Connections with missing neurotransmitter: {missing}")

# Save as Parquet
neurons.write_parquet("data/banc_neurons_v888.parquet")
connections.write_parquet("data/banc_connections_v888.parquet")

print(neurons.head())
print(connections.head())
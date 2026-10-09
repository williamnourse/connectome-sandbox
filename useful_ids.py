import polars as pl

def get_ids(dataframe):
    return dataframe.select(pl.col("Root ID")).to_series().to_list()

def __check_leg__(leg: str):
    if leg not in ["front", "middle", "hind"]:
        raise ValueError("leg must be one of 'front', 'middle', or 'hind'")

def __check_side__(side: str):
    if side not in ["left", "right"]:
        raise ValueError("side must be one of 'left' or 'right'")

def get_motor_neurons(leg: str, side: str):
    __check_leg__(leg)
    __check_side__(side)
    neurons = pl.read_parquet("data/banc_neurons_v888.parquet")
    all_mns = neurons.filter(pl.col("Class") == "leg_motor_neuron")
    leg_mns = all_mns.filter(pl.col("Sub Class") == f"{leg}_leg_motor_neuron")
    specific_mns = leg_mns.filter(pl.col("Soma side") == side)
    return specific_mns

def get_descending_neurons(side: str):
    __check_side__(side)
    if side == "left":
        other_side = "right"
    else:
        other_side = "left"
    neurons = pl.read_parquet("data/banc_neurons_v888.parquet")
    DNg100 = neurons.filter(pl.col("Primary Cell Type") == "DNg100", pl.col("Soma side") == other_side)
    DNg97 = neurons.filter(pl.col("Primary Cell Type") == "DNg97", pl.col("Soma side") == other_side)
    descending_mns = pl.concat([DNg100, DNg97], how="vertical")
    return descending_mns

def get_cpg_neurons():
    neurons = pl.read_parquet("data/banc_neurons_v888.parquet")
    IN03A006 = neurons.filter(pl.col("Root ID") == 720575941354124592)
    INXXX464 = neurons.filter(pl.col("Root ID") == 720575941603886582)
    IN12B003 = neurons.filter(pl.col("Root ID") == 720575941438748607)
    IN17A001 = neurons.filter(pl.col("Root ID") == 720575941554038555)
    IN09A002 = neurons.filter(pl.col("Root ID") == 720575941623146186)
    cpg_mns = pl.concat([IN03A006, INXXX464, IN12B003, IN17A001, IN09A002], how="vertical")
    return cpg_mns

def get_vnc_neurons(side: str):
    __check_side__(side)
    neurons = pl.read_parquet("data/banc_neurons_v888.parquet")
    VNC_hemicord = neurons.filter(pl.col("Soma side") == side, pl.col("Community labels").str.contains('soma in VNC'))
    return VNC_hemicord

if __name__ == "__main__":
    leg = 'front'
    side = 'right'
    front_right_mns = get_motor_neurons(leg, side)
    print('Front Right Motor Neurons:')
    print(front_right_mns)
    descending_mns = get_descending_neurons(side)
    print('Descending Neurons:')
    print(descending_mns)
    cpg = get_cpg_neurons()
    print('CPG Neurons:')
    print(cpg)
    vnc_hemicord = get_vnc_neurons(side)
    print('VNC Neurons:')
    print(vnc_hemicord)